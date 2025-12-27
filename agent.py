import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3, **simulate_kwargs):
    """带超时保护的物理模拟。

    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
        **simulate_kwargs: 透传给 pooltool.pt.simulate 的参数（如 max_events 等）

    返回：
        bool: True 表示模拟成功，False 表示超时

    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）。
        超时后自动恢复，不会导致程序卡死。
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        # 允许传入 max_events 等参数（NewAgent 的抗噪评估依赖该能力）
        # 强制 inplace=True；若外部也传入 inplace，则以这里为准。
        simulate_kwargs.pop('inplace', None)
        pt.simulate(shot, inplace=True, **simulate_kwargs)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        signal.alarm(0)  # 取消超时
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False

        # 大量打印会显著拖慢评测；可通过环境变量开启：BILLIARDS_DEBUG=1
        self.debug_mode = bool(int(os.getenv("BILLIARDS_DEBUG", "0")))

        if self.debug_mode:
            print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            if self.debug_mode:
                print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                if self.debug_mode:
                    print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            if self.debug_mode:
                print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                if self.debug_mode:
                    print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            if self.debug_mode:
                print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                    f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                    f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            if self.debug_mode:
                print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class NewAgent(Agent):
    """NewAgent - Phase 25: Ultra Safe Edition

    核心改进：
    1. 提高犯规惩罚权重（-5000起步）
    2. 强制三重验证（不允许跳过）
    3. 增加抗噪测试到10次
    4. 添加调试日志定位问题
    """
    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        self.SEARCH_INIT = 12  # 降低搜索次数，提高质量
        self.SEARCH_ITER = 8

        # a/b 的极值会显著增加“偏杆导致的偏转/不确定性”，进而抬高
        # no-hit / first-contact / scratch 的概率。环境允许到 0.5，但这里更保守。
        self.AB_LIMIT = 0.25
        
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 
            'a': 0.003, 'b': 0.003
        }
        
        # 调试模式：默认关闭（大量打印会显著拖慢120局评测）
        # 可通过环境变量开启：BILLIARDS_DEBUG=1
        self.debug_mode = bool(int(os.getenv("BILLIARDS_DEBUG", "0")))
        
        print("[NewAgent] Phase 25: Ultra Safe Edition 已初始化")
        print("[提示] 如果还有高犯规，请检查 evaluation_log.json 找出具体原因")

    # ========== 工具函数 ==========
    def _distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

    def _ray_point_distance_2d(self, ray_origin, ray_dir, point):
        """Return (perp_distance, t) from a ray to a point in 2D.

        ray_origin: array-like (2,)
        ray_dir: array-like (2,) assumed normalized
        point: array-like (2,)
        t: projection length along ray_dir (t<0 means behind origin)
        """
        o = np.asarray(ray_origin, dtype=float)
        d = np.asarray(ray_dir, dtype=float)
        p = np.asarray(point, dtype=float)
        v = p - o
        t = float(np.dot(v, d))
        closest = o + t * d
        dist = float(np.linalg.norm(p - closest))
        return dist, t

    def _estimate_scratch_risk(self, cue_pos, ghost_pos, table):
        """Heuristic scratch-risk estimate.

        Approximates cue-ball post-impact direction as cue->ghost ray and penalizes
        cases where that ray passes close to any pocket mouth.
        """
        c = np.asarray(cue_pos[:2], dtype=float)
        g = np.asarray(ghost_pos[:2], dtype=float)
        d = self._normalize(g - c)

        # Tuned conservatively: only penalize when the aim ray runs very close to a pocket.
        # This complements (not replaces) Monte Carlo safety checks.
        near_threshold = 0.11  # meters
        min_dist = 1e9
        min_t = 0.0
        for _, pocket in table.pockets.items():
            p = np.asarray(pocket.center[:2], dtype=float)
            dist, t = self._ray_point_distance_2d(c, d, p)
            # Only consider pockets in front of the cue ball (ignore behind).
            if t <= 0.15:
                continue
            if dist < min_dist:
                min_dist = dist
                min_t = t

        # No meaningful pocket in front
        if min_dist >= 1e8:
            return 0.0

        # Risk ramps up when very close to a pocket mouth.
        if min_dist >= near_threshold:
            return 0.0

        # Slightly down-weight extremely far pockets to avoid over-penalizing long rays.
        distance_weight = 1.0 if min_t < 1.2 else 0.7
        return float(distance_weight * (near_threshold - min_dist) / near_threshold)

    def _normalize(self, vec):
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])

    def _angle_to_phi(self, direction_vec):
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360

    def _calculate_ghost_ball(self, target_pos, pocket_pos):
        target_to_pocket = self._normalize(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        return np.array(target_pos[:2]) - target_to_pocket * (2 * self.BALL_RADIUS)

    def _calculate_cut_angle(self, cue_pos, target_pos, pocket_pos):
        ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
        vec1 = self._normalize(np.array(ghost_pos) - np.array(cue_pos[:2]))
        vec2 = self._normalize(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        count = 0
        line_vec = np.array(to_pos[:2]) - np.array(from_pos[:2])
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-6: return 0
        line_dir = line_vec / line_length
        
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4: continue
            ball_pos = ball.state.rvw[0][:2]
            vec_to_ball = ball_pos - np.array(from_pos[:2])
            proj_length = np.dot(vec_to_ball, line_dir)
            if proj_length < 0 or proj_length > line_length: continue
            proj_point = np.array(from_pos[:2]) + line_dir * proj_length
            dist_to_line = np.linalg.norm(ball_pos - proj_point)
            if dist_to_line < self.BALL_RADIUS * 2.2: count += 1
        return count

    # ========== 严格清台判断 ==========
    def _check_can_shoot_8(self, balls, my_targets):
        """判断是否可以打黑8"""
        real_targets = [bid for bid in my_targets if bid != '8']
        remaining = [bid for bid in real_targets if balls[bid].state.s != 4]
        can_shoot = len(remaining) == 0
        
        if self.debug_mode and can_shoot:
            print(f"   [DEBUG] ✅ 己方球已清空，现在可以打黑8")
        
        return can_shoot

    def _get_valid_targets(self, balls, my_targets):
        """获取当前合法目标（严格区分清台前后）"""
        can_shoot_8 = self._check_can_shoot_8(balls, my_targets)
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if can_shoot_8:
            valid = ['8']
            if self.debug_mode:
                print(f"   [DEBUG] 合法目标：['8'] (清台后)")
        else:
            # 绝对不能包含黑8！
            valid = [bid for bid in remaining if bid != '8']
            if self.debug_mode and valid:
                print(f"   [DEBUG] 合法目标：{valid[:3]}... (清台前，共{len(valid)}个)")
        
        return valid, can_shoot_8

    # ========== 首球碰撞检测 ==========
    def _check_first_contact(self, shot, valid_target_ids):
        """检测首球碰撞是否合法"""
        valid_ball_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            
            if 'cushion' in et or 'pocket' in et:
                continue
            
            if 'cue' in ids:
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_ball = other_ids[0]
                    is_legal = (first_ball in valid_target_ids)
                    return is_legal, first_ball
        
        return False, None

    def _is_eight_in_pocket_danger(self, balls, table, threshold: float = 0.22) -> bool:
        """Return True if the 8-ball is already close to any pocket mouth.

        When the 8 is near a pocket pre-clear, even light incidental contact can
        turn into an illegal-8 pocket under execution noise.
        """
        try:
            if '8' not in balls:
                return False
            if balls['8'].state.s == 4:
                return False
            eight_xy = np.asarray(balls['8'].state.rvw[0][:2], dtype=float)
            pocket_centers = [np.asarray(p.center[:2], dtype=float) for p in table.pockets.values()]
            if not pocket_centers:
                return False
            min_dist = min(float(np.linalg.norm(eight_xy - pc)) for pc in pocket_centers)
            return min_dist < float(threshold)
        except Exception:
            return False

    def _shot_touches_ball(self, shot, ball_id: str) -> bool:
        """Return True if any non-pocket/non-cushion event involves `ball_id`."""
        valid_ball_ids = {
            '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15',
            'cue',
        }

        for e in getattr(shot, 'events', []) or []:
            et = str(getattr(e, 'event_type', '')).lower()
            if 'cushion' in et or 'pocket' in et:
                continue
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ball_id in ids and any((i != ball_id) and (i in valid_ball_ids) for i in ids):
                return True

        return False

    def _sanitize_action(self, action):
        """标准化/裁剪动作参数，避免极端 a/b 导致的高风险出杆。"""
        if action is None:
            return None
        out = dict(action)
        out['V0'] = float(np.clip(out.get('V0', 2.5), 0.5, 8.0))
        out['phi'] = float(out.get('phi', 0.0) % 360)
        out['theta'] = float(np.clip(out.get('theta', 0.0), 0.0, 90.0))
        out['a'] = float(np.clip(out.get('a', 0.0), -self.AB_LIMIT, self.AB_LIMIT))
        out['b'] = float(np.clip(out.get('b', 0.0), -self.AB_LIMIT, self.AB_LIMIT))
        return out

    # ========== 核心：全面安全检查（提高到10次） ==========
    def _simulate_envlike(self, shot, timeout=3):
        """尽量对齐环境：使用较高的 max_events + 超时保护。

        说明：PoolEnv 侧是 `pt.simulate(..., inplace=True)`（不显式限制 max_events）。
        这里完全不设上限会显著变慢并导致超时过多，反而让严格安全门失效。
        采用“较高但有限”的 max_events 来覆盖绝大多数后续进袋/碰撞事件。
        """
        ok = simulate_with_timeout(shot, timeout=timeout, max_events=1200)
        if not ok:
            raise SimulationTimeoutError("simulate timeout")

    def _assess_action_risk(self, action, balls, table, valid_targets, simulations=10):
        """在噪声下评估动作风险，返回每类致命/犯规事件的出现次数。

        目的：
        - 比 bool 更可解释（能定位是 scratch / first-contact / no-hit 等哪类在冒头）
        - 使用 env-like 仿真（无 max_events 截断），避免漏掉后续进袋导致的误判
        """
        can_shoot_8 = ('8' in valid_targets)
        action = self._sanitize_action(action)

        eight_start_xy = None
        pocket_centers_xy = None
        if table is not None:
            try:
                pocket_centers_xy = [np.asarray(p.center[:2], dtype=float) for p in table.pockets.values()]
            except Exception:
                pocket_centers_xy = None

        if (not can_shoot_8) and ('8' in balls):
            try:
                eight_start_xy = np.asarray(balls['8'].state.rvw[0][:2], dtype=float)
            except Exception:
                eight_start_xy = None

        risk = {
            'physics_fail': 0,
            'cue_pocket': 0,
            'eight_illegal': 0,
            'first_contact': 0,
            'no_hit': 0,
            'no_pocket_no_rail': 0,
            # Pre-clear diagnostics (not necessarily fatal): used to trigger extra MC
            'eight_moved': 0,
            'eight_near_pocket': 0,
            'eight_danger': 0,
        }

        # 未清台时提高上限，优先捕捉“误打黑8”等后续事件
        sim_max_events = 1400 if (not can_shoot_8) else 1000

        for _ in range(int(simulations)):
            noisy_action = {
                'V0': float(np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0)),
                'phi': float((action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360),
                'theta': float(np.clip(action['theta'] + np.random.normal(0, self.noise_std['theta']), 0, 90)),
                'a': float(np.clip(action['a'] + np.random.normal(0, self.noise_std['a']), -self.AB_LIMIT, self.AB_LIMIT)),
                'b': float(np.clip(action['b'] + np.random.normal(0, self.noise_std['b']), -self.AB_LIMIT, self.AB_LIMIT)),
            }

            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            try:
                shot.cue.set_state(**noisy_action)
                ok = simulate_with_timeout(shot, timeout=3, max_events=sim_max_events)
                if not ok:
                    raise SimulationTimeoutError("simulate timeout")
            except Exception:
                risk['physics_fail'] += 1
                continue

            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
            if 'cue' in new_pocketed:
                risk['cue_pocket'] += 1
                continue
            if ('8' in new_pocketed) and (not can_shoot_8):
                risk['eight_illegal'] += 1
                continue

            # Pre-clear: if the 8-ball gets disturbed (moves) or ends up near a pocket,
            # we treat this shot as "tail-risky" and will trigger extra MC sampling.
            if (not can_shoot_8) and eight_start_xy is not None and pocket_centers_xy:
                try:
                    eight_end_xy = np.asarray(shot.balls['8'].state.rvw[0][:2], dtype=float)
                    # Ignore tiny numerical drift; only flag meaningful 8 motion.
                    if float(np.linalg.norm(eight_end_xy - eight_start_xy)) > 0.02:
                        risk['eight_moved'] += 1
                    min_pocket_dist = min(float(np.linalg.norm(eight_end_xy - pc)) for pc in pocket_centers_xy)
                    if min_pocket_dist < 0.17:
                        risk['eight_near_pocket'] += 1
                    if min_pocket_dist < 0.25:
                        risk['eight_danger'] += 1
                except Exception:
                    pass

            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                risk['first_contact'] += 1
                if first_ball is None:
                    risk['no_hit'] += 1
                continue

            # 若无进球，则必须满足碰库规则（对齐 poolenv）
            if len(new_pocketed) == 0 and first_ball is not None:
                cue_hit_cushion = False
                target_hit_cushion = False
                for e in shot.events:
                    et = str(e.event_type).lower()
                    ids = list(e.ids) if hasattr(e, 'ids') else []
                    if 'cushion' in et:
                        if 'cue' in ids:
                            cue_hit_cushion = True
                        if first_ball in ids:
                            target_hit_cushion = True
                if (not cue_hit_cushion) and (not target_hit_cushion):
                    risk['no_pocket_no_rail'] += 1

        return risk

    def _pick_min_risk_action(self, actions, balls, table, my_targets, simulations=6):
        """在候选动作中选“风险最小”的一个，用于 strict gate 下的最后兜底。

        排序目标（从最重要到次要）：
        1) first_contact + no_hit（你选择的A策略优先压）
        2) cue_pocket
        3) eight_illegal
        4) no_pocket_no_rail
        5) physics_fail
        """
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return None

        best = None
        best_key = None
        for a in actions:
            if a is None:
                continue
            risk = self._assess_action_risk(a, balls, table, valid_targets, simulations=simulations)
            key = (
                risk['first_contact'] + risk['no_hit'],
                risk['cue_pocket'],
                risk['eight_illegal'],
                risk['no_pocket_no_rail'],
                risk['physics_fail'],
            )
            if best_key is None or key < best_key:
                best_key = key
                best = a

        if self.debug_mode and best_key is not None:
            print(f"   [DEBUG] min-risk key={best_key}")
        return best

    def _is_action_safe(self, action, balls, table, valid_targets, simulations=10):
        """严格安全判定：关键犯规在噪声测试中出现 1 次也拒绝。"""
        risk = self._assess_action_risk(action, balls, table, valid_targets, simulations=simulations)

        # Phase A: 以“强压犯规”为优先目标
        # - cue_pocket/eight_illegal/first_contact/no_hit: 严格 0 容忍
        # - no_pocket_no_rail: 在噪声下可能偶发（尤其是轻球），允许极小概率，避免过度保守
        rail_allow = 1 if int(simulations) >= 8 else 0
        ok = (
            risk['physics_fail'] == 0
            and risk['cue_pocket'] == 0
            and risk['eight_illegal'] == 0
            and risk['first_contact'] == 0
            and risk['no_hit'] == 0
            and risk['no_pocket_no_rail'] <= rail_allow
        )

        # Adaptive tightening: pre-clear illegal-8 is game-losing but can be a low-probability tail event.
        # If we see warning signs, re-check with more samples (kept narrow to control runtime).
        can_shoot_8 = ('8' in valid_targets)
        need_extra = (
            (not can_shoot_8)
            and (
                risk.get('eight_near_pocket', 0) > 0
                or (risk.get('eight_moved', 0) > 0 and risk.get('eight_danger', 0) > 0)
            )
        )

        if ok and int(simulations) < 18 and need_extra:
            risk2 = self._assess_action_risk(action, balls, table, valid_targets, simulations=18)
            rail_allow2 = 1  # >=8
            ok = (
                risk2['physics_fail'] == 0
                and risk2['cue_pocket'] == 0
                and risk2['eight_illegal'] == 0
                and risk2['first_contact'] == 0
                and risk2['no_hit'] == 0
                and risk2['no_pocket_no_rail'] <= rail_allow2
            )

        if (not ok) and self.debug_mode:
            print(f"   [DEBUG] risk={risk}")
        return ok

    def _simulate_deterministic_once(self, action, balls, table, max_events=550):
        """无噪声确定性仿真一次，用于快速判定“是否至少是合法且能碰到球”。"""
        action = self._sanitize_action(action)
        sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
        shot.cue.set_state(**action)
        # 用超时保护 + 更高的 max_events，避免漏掉“后续黑8进袋/白球进袋”等长尾事件。
        ok = simulate_with_timeout(shot, timeout=2, max_events=max_events)
        if not ok:
            raise SimulationTimeoutError("simulate timeout")
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
        return shot, new_pocketed

    def _is_action_legal_deterministic(self, action, balls, table, valid_targets):
        """确定性合法性检查：避免 no_hit / 首球犯规 / 白球进袋 / 误打黑8。"""
        can_shoot_8 = ('8' in valid_targets)
        eight_danger = (not can_shoot_8) and self._is_eight_in_pocket_danger(balls, table)
        try:
            # 清台前提高 max_events，减少“误打黑8/白球进袋”被截断漏检
            det_events = 950 if (not can_shoot_8) else 550
            shot, new_pocketed = self._simulate_deterministic_once(action, balls, table, max_events=det_events)
        except Exception:
            return False

        if 'cue' in new_pocketed:
            return False
        if '8' in new_pocketed and not can_shoot_8:
            return False
        # Conditional pre-clear constraint: only forbid 8 contact when 8 is already near a pocket.
        if eight_danger and self._shot_touches_ball(shot, '8'):
            return False

        is_legal, first_ball = self._check_first_contact(shot, valid_targets)
        if not is_legal:
            return False

        # 若未进球，仍需满足“碰库”规则
        if len(new_pocketed) == 0 and first_ball is not None:
            cue_hit_cushion = False
            target_hit_cushion = False
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                if 'cushion' in et:
                    if 'cue' in ids:
                        cue_hit_cushion = True
                    if first_ball in ids:
                        target_hit_cushion = True
            if not cue_hit_cushion and not target_hit_cushion:
                return False

        return True

    def _try_repair_action(self, action, balls, table, my_targets, safety_sims=8):
        """在动作不安全时，做小范围的(v0,phi)修补搜索，优先降低白球进袋/误打黑8/首球犯规风险。"""
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return None

        # 修补策略：优先降速，其次微调角度
        v_scales = [0.95, 0.9, 0.85, 0.8, 0.75]
        phi_offsets = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]

        base = self._sanitize_action(action)
        # 额外保守：如果速度很大，先限制到6.5以内再尝试
        base['V0'] = float(np.clip(base.get('V0', 3.0), 0.5, 6.5))

        for vs in v_scales:
            for dphi in phi_offsets:
                cand = base.copy()
                cand['V0'] = float(np.clip(base['V0'] * vs, 0.5, 8.0))
                cand['phi'] = float((base['phi'] + dphi) % 360)
                # 先过确定性合法性，避免把“根本碰不到球”的候选送进昂贵MC
                if not self._is_action_legal_deterministic(cand, balls, table, valid_targets):
                    continue
                if self._is_action_safe(cand, balls, table, valid_targets, simulations=safety_sims):
                    return cand
        return None

    def _find_any_safe_action(self, balls, table, my_targets, attempts=36, safety_sims=6):
        """兜底：尝试构造任意一个安全动作（严格：必须通过安全验证）。"""
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return None
        cue_pos = balls['cue'].state.rvw[0]

        # 在合法目标中优先选离白球近的，降低大角度/大力度需求
        target_ids = [tid for tid in valid_targets if tid in balls and balls[tid].state.s != 4]
        target_ids.sort(key=lambda tid: self._distance(cue_pos, balls[tid].state.rvw[0]))
        if not target_ids:
            return None

        # strict gate 下需要更“可行”的探索空间，否则容易找不到任何可用动作
        phi_jitter = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0, 8.0, -8.0, 12.0, -12.0]
        v_candidates = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

        tries = 0
        for tid in target_ids[:4]:
            vec = balls[tid].state.rvw[0] - cue_pos
            base_phi = self._angle_to_phi(self._normalize(vec))
            dist = self._distance(cue_pos, balls[tid].state.rvw[0])
            for v0 in v_candidates:
                for dphi in phi_jitter:
                    tries += 1
                    if tries > attempts:
                        return None
                    action = {
                        'V0': float(np.clip(v0 + 0.3 * dist, 2.2, 5.0)),
                        'phi': float((base_phi + dphi) % 360),
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0,
                    }
                    if not self._is_action_legal_deterministic(action, balls, table, valid_targets):
                        continue
                    if self._is_action_safe(action, balls, table, valid_targets, simulations=safety_sims):
                        return action
        return None

    def _finalize_action(self, action, balls, table, my_targets, safety_sims=10):
        """统一出口：保证返回的动作尽可能安全；不安全则修补/兜底。"""
        action = self._sanitize_action(action)
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)

        # 清台前把安全验证加严，专门压“误打黑8/白球进袋/首球犯规”的尾部风险
        stage_sims = max(int(safety_sims), 14) if (not can_shoot_8) else int(safety_sims)
        # 先做一次确定性合法性预检，避免把“根本碰不到球/首球不合法”的动作送进昂贵 MC。
        if valid_targets and (not self._is_action_legal_deterministic(action, balls, table, valid_targets)):
            repaired0 = self._try_repair_action(action, balls, table, my_targets, safety_sims=max(6, stage_sims - 2))
            if repaired0 is not None:
                action = repaired0

        if valid_targets and self._is_action_safe(action, balls, table, valid_targets, simulations=stage_sims):
            return action

        repaired = self._try_repair_action(action, balls, table, my_targets, safety_sims=max(6, stage_sims - 2))
        if repaired is not None:
            return repaired

        fallback = self._find_any_safe_action(balls, table, my_targets, attempts=120, safety_sims=6)
        if fallback is not None:
            return fallback

        # strict gate 可能导致“完全找不到安全动作”，此时不能直接返回原动作（会抬高犯规）。
        # 改为：构造一组“保证能碰到合法球”的保守候选，并选择风险最小的一个。
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        cue_pos = balls['cue'].state.rvw[0]
        target_ids = [tid for tid in valid_targets if tid in balls and balls[tid].state.s != 4]
        if target_ids:
            target_ids.sort(key=lambda tid: self._distance(cue_pos, balls[tid].state.rvw[0]))
            tid0 = target_ids[0]
            base_phi = self._angle_to_phi(self._normalize(balls[tid0].state.rvw[0] - cue_pos))
            phi_offsets = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0]
            v0s = [2.5, 3.0, 3.5, 4.0, 4.5]
            candidates = []
            for v0 in v0s:
                for dphi in phi_offsets:
                    cand = {
                        'V0': float(v0),
                        'phi': float((base_phi + dphi) % 360),
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0,
                    }
                    if self._is_action_legal_deterministic(cand, balls, table, valid_targets):
                        candidates.append(cand)

            # 先尝试找一个真正 safe 的（用较少次数提高可找到性）
            for cand in candidates:
                if self._is_action_safe(cand, balls, table, valid_targets, simulations=6):
                    return cand

            # 否则选“最小风险”的一个，尽量压 first-contact/no-hit
            picked = self._pick_min_risk_action(candidates, balls, table, my_targets, simulations=6)
            if picked is not None:
                return picked

        # 放宽标准的最后一试：只在“允许打8”的收官阶段启用。
        # 清台前放宽会显著抬高 eight_illegal（误打黑8）。
        if can_shoot_8:
            relaxed_sims = 3
            if valid_targets and self._is_action_safe(action, balls, table, valid_targets, simulations=relaxed_sims):
                return action

            repaired_relaxed = self._try_repair_action(action, balls, table, my_targets, safety_sims=relaxed_sims)
            if repaired_relaxed is not None:
                return repaired_relaxed

            fallback_relaxed = self._find_any_safe_action(balls, table, my_targets, attempts=70, safety_sims=relaxed_sims)
            if fallback_relaxed is not None:
                return fallback_relaxed

        # 最后兜底：返回原动作（极少发生）
        return action

    # ========== 贝叶斯优化（提高惩罚） ==========
    def _optimized_search(self, geo_action, balls, my_targets, table):
        """贝叶斯优化 + 超严格惩罚"""
        geo_action = self._sanitize_action(geo_action)
        pbounds = {
            'V0': (max(0.5, geo_action['V0'] - 1.5), min(8.0, geo_action['V0'] + 1.5)),
            'phi': (geo_action['phi'] - 3.0, geo_action['phi'] + 3.0),  # 扩大搜索范围
            # 稳定性优先：固定 theta/a/b，减少因偏杆/跳球带来的噪声敏感与白球进袋
            'theta': (0, 0), 'a': (0.0, 0.0), 'b': (0.0, 0.0)
        }
        
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        
        if not valid_targets:
            print("   [WARNING] 无有效目标球")
            return geo_action

        def reward_fn(V0, phi, theta, a, b):
            # theta/a/b 已固定；这里仍做一次明确赋值，避免边界浮动
            theta = 0.0
            a = 0.0
            b = 0.0
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            
            try:
                shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                return -5000  # 提高惩罚
            
            # 死锁检测
            for ball in shot.balls.values():
                if ball.state.s not in [0, 4]:
                    return -10000  # 提高惩罚
            
            # 进袋检测（超严格惩罚）
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and last_state[bid].state.s != 4]
            
            if 'cue' in new_pocketed:
                return -5000  # -1000 → -5000
            
            if '8' in new_pocketed and not can_shoot_8:
                return -10000  # -1500 → -10000
            
            # 首球犯规检测（严格惩罚）
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                return -3000  # -800 → -3000
            
            # 基础分
            base_score = analyze_shot_for_reward(shot, last_state, valid_targets)
            
            # 走位奖励
            own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
            position_bonus = 0
            if len(own_pocketed) > 0 and '8' not in new_pocketed:
                final_cue_pos = shot.balls['cue'].state.rvw[0]
                pos_quality = self._evaluate_position_quality(
                    final_cue_pos, shot.balls, my_targets, my_targets
                )
                position_bonus = pos_quality * 20  # 降低走位权重，优先安全
            
            return base_score + position_bonus

        try:
            optimizer = BayesianOptimization(
                f=reward_fn, pbounds=pbounds, random_state=42, verbose=0
            )
            optimizer.maximize(init_points=self.SEARCH_INIT, n_iter=self.SEARCH_ITER)
            
            # 提高接受阈值
            if optimizer.max['target'] > 0:  # 原来是-100，改为0
                p = optimizer.max['params']
                return {
                    'V0': p['V0'], 'phi': p['phi'], 'theta': p['theta'], 
                    'a': p['a'], 'b': p['b']
                }
            else:
                if self.debug_mode:
                    print(f"   [DEBUG] 优化器最高分={optimizer.max['target']:.1f}，低于阈值")
        except Exception as e:
            print(f"   [ERROR] 优化失败: {e}")
        
        return geo_action

    # ========== 验证阶段（强制三重检查）==========
    def _validate_and_adjust(self, action, balls, table, my_targets):
        """验证并微调（不允许跳过安全检查）"""
        action = self._sanitize_action(action)
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        eight_danger = (not can_shoot_8) and self._is_eight_in_pocket_danger(balls, table)
        
        if not valid_targets:
            return self._defense_shot(balls, my_targets, table)
        
        variations = [
            (1.0, 0), (0.95, 0), (1.05, 0),
            (1.0, 0.5), (1.0, -0.5), (1.0, 1.0), (1.0, -1.0),
            (0.9, 0), (1.1, 0)  # 新增更大变化
        ]
        
        best_safe_action = None
        best_with_pocket = None
        
        for v_scale, phi_offset in variations:
            test_action = action.copy()
            test_action['V0'] = np.clip(test_action['V0'] * v_scale, 0.5, 8.0)
            test_action['phi'] = (test_action['phi'] + phi_offset) % 360
            
            # === 第一步：快速确定性测试 ===
            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            
            try:
                shot.cue.set_state(**test_action)
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                continue
            
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            # 基础过滤
            if 'cue' in new_pocketed or ('8' in new_pocketed and not can_shoot_8):
                continue

            if eight_danger and self._shot_touches_ball(shot, '8'):
                continue
            
            is_legal, _ = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                continue
            
            # === 第二步：抗噪鲁棒性测试（强制执行，不允许跳过）===
            # 清台前对“误打黑8/白球进袋”更敏感，增加测试次数以压低尾部风险
            sims = 14 if (not can_shoot_8) else 10
            if not self._is_action_safe(test_action, balls, table, valid_targets, simulations=sims):
                continue  # 🔴 关键：必须通过安全测试
            
            # === 第三步：优先返回进球方案 ===
            own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
            if len(own_pocketed) > 0:
                best_with_pocket = test_action
                # 找到“安全+进球”就立即返回，避免继续做昂贵的安全测试
                break
            
            # 记录安全的无进球方案（只要找到一个就可作为兜底）
            if best_safe_action is None:
                best_safe_action = test_action
        
        if best_with_pocket:
            if self.debug_mode:
                print("   [DEBUG] ✅ 找到安全+进球方案")
            return best_with_pocket
        
        if best_safe_action:
            if self.debug_mode:
                print("   [DEBUG] ⚠️ 仅找到安全方案（无进球）")
            return best_safe_action
        
        # 兜底防守
        print("   [PROTECTOR] 🛡️ 所有进攻路线不安全，启动防守")
        return self._defense_shot(balls, my_targets, table)

    # ========== 辅助函数 ==========
    def _evaluate_position_quality(self, cue_pos, balls, my_targets, original_targets):
        """评估白球位置"""
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return 1.0
        
        min_dist = 100.0
        for tid in valid_targets:
            if balls[tid].state.s == 4: continue
            dist = self._distance(cue_pos, balls[tid].state.rvw[0])
            if dist < min_dist: min_dist = dist
        
        return 1.0 if (0.2 < min_dist < 1.0) else 0.5

    def get_break_shot(self, balls):
        """开球（已修复：必须首球合法）"""
        # 保留旧签名以兼容外部调用；实际在 decision() 中会走 get_break_shot_for_targets
        cue = balls['cue']
        target = balls['1']
        vec = target.state.rvw[0] - cue.state.rvw[0]
        phi = self._angle_to_phi(self._normalize(vec))
        return {'V0': 8.0, 'phi': phi, 'theta': 0, 'a': 0.0, 'b': 0.0}

    def get_break_shot_for_targets(self, balls, my_targets, table):
        """开球：用仿真搜索一个“首碰合法”的动作。

        背景：本项目的规则是“白球必须先接触己方目标球”，这会导致传统的
        “总是冲 1 号球”的开球在 PlayerA=stripe 时几乎必犯规。
        
        策略：围绕球堆中心方向采样若干 (phi, V0)，对每个候选做一次确定性仿真，
        只保留：不白球进袋、不误打黑8、且首碰为合法目标球。
        """
        cue_pos = balls['cue'].state.rvw[0]
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return {'V0': 8.0, 'phi': 0.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0}

        # 球堆中心方向（用于生成phi基准）
        rack_positions = [b.state.rvw[0] for bid, b in balls.items() if bid != 'cue' and b.state.s != 4]
        if rack_positions:
            rack_center = np.mean(np.asarray(rack_positions), axis=0)
            base_phi = self._angle_to_phi(self._normalize(rack_center - cue_pos))
        else:
            base_phi = 0.0

        # 为避免 stripe 开球“几何上不可直达”的情况，直接在phi上做更宽的搜索
        phi_offsets = list(range(-35, 36, 5))  # -35..35 step 5
        # 增加低速候选，显著降低噪声下的白球进袋风险（开球只发生一次/局，稍慢可接受）
        v0_candidates = [8.0, 7.0, 6.5, 6.0, 5.5, 5.0]

        def simulate_once(action):
            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            shot.cue.set_state(**action)
            pt.simulate(shot, inplace=True, max_events=400)
            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
            return shot, new_pocketed

        candidates = []
        for dphi in phi_offsets:
            for v0 in v0_candidates:
                action = {
                    'V0': float(v0),
                    'phi': float((base_phi + dphi) % 360),
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0,
                }
                try:
                    shot, new_pocketed = simulate_once(action)
                except Exception:
                    continue

                # 致命事件过滤
                if 'cue' in new_pocketed:
                    continue
                if ('8' in new_pocketed) and (not can_shoot_8):
                    continue

                is_legal, first_ball = self._check_first_contact(shot, valid_targets)
                if not is_legal:
                    continue

                own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
                enemy_pocketed = [bid for bid in new_pocketed if bid not in valid_targets and bid not in ['cue', '8']]
                score = 100 * len(own_pocketed) - 50 * len(enemy_pocketed)
                # 首球信息用于调试
                candidates.append((score, len(own_pocketed), -len(enemy_pocketed), action))

        # 优先选择“能进己方球”的合法开球；否则选任意合法开球
        # NOTE: candidates contains a dict `action` as the last tuple item; if earlier
        # score components tie, Python would try to compare dicts (TypeError).
        candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        for _, _, _, action in candidates[:12]:
            # 开球动作数量很少：用更强的抗噪验证，优先压白球进袋/首球犯规
            if self._is_action_safe(action, balls, table, valid_targets, simulations=12):
                return action

        # 若都没通过抗噪验证，但存在确定性合法开球，仍优先返回最高分
        if candidates:
            return candidates[0][3]

        # 最终兜底：朝球堆中心打
        return {'V0': 6.0, 'phi': float(base_phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0}

    def _choose_best_target(self, balls, my_targets, table):
        """选择最佳目标（返回单个最优候选）。"""
        cands = self._choose_top_targets(balls, my_targets, table, top_k=1)
        return (cands[0][0], cands[0][1]) if cands else None

    def _choose_top_targets(self, balls, my_targets, table, top_k=6):
        """枚举并返回多个“直塞进袋”候选。

        设计目标：
        - 过去只选一个(目标,袋口)容易卡死在“看起来最好但很难进”的路线。
        - 这里先用几何/遮挡启发式做粗筛，再在 decision() 里对 top-K 做更精确评估。

        Returns:
            list[tuple[target_id, pocket_id, heuristic_score]] (按分数降序)
        """
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return []

        cue_pos = balls['cue'].state.rvw[0]
        candidates = []

        for target_id in valid_targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 粗筛：切角过大直接放弃（略放宽以增加可用进攻线路）
                cut_angle = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos)
                if cut_angle > 80:
                    continue

                # 粗筛：直线遮挡直接放弃（不做借库/跳球）
                obs_1 = self._count_obstructions(balls, cue_pos, target_pos, exclude_ids=['cue', target_id])
                if obs_1 > 0:
                    continue
                obs_2 = self._count_obstructions(balls, target_pos, pocket_pos, exclude_ids=['cue', target_id])
                if obs_2 > 0:
                    continue

                # 启发式打分（只用于排序，不代表最终价值）
                dist_cue_target = self._distance(cue_pos, target_pos)
                dist_target_pocket = self._distance(target_pos, pocket_pos)
                score = 0.0
                score += 50.0 / (1.0 + dist_cue_target + dist_target_pocket)
                score += (90.0 - cut_angle) * 1.5

                # ghost 附近的“误入其它袋”风险
                ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
                for _, p_danger in table.pockets.items():
                    if self._distance(ghost_pos, p_danger.center) < 0.15:
                        score -= 400.0

                # 轻量 scratch 风险（仅当作 tie-breaker）
                scratch_risk = self._estimate_scratch_risk(cue_pos, np.append(ghost_pos, 0.0), table)
                if scratch_risk > 0:
                    score -= 250.0 * scratch_risk

                # 收官阶段：鼓励黑8
                if target_id == '8' and can_shoot_8:
                    score += 500.0

                candidates.append((target_id, pocket_id, float(score)))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[: max(1, int(top_k))]

    def _quick_shot_quality(self, action, balls, table, my_targets):
        """对动作做一次无噪声仿真，并给出“进攻价值”粗评分。

        用途：在多个候选之间做决策，不替代 finalize 的严格安全检查。

        Returns:
            (quality_score, own_pocketed_count, enemy_pocketed_count)
        """
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        if not valid_targets:
            return -1e9, 0, 0

        # 清台前提高 max_events，减少误判（尤其是后续黑8/白球进袋）
        max_events = 950 if (not can_shoot_8) else 450
        try:
            shot, new_pocketed = self._simulate_deterministic_once(action, balls, table, max_events=max_events)
        except Exception:
            return -1e9, 0, 0

        # 与环境对齐：白球/非法黑8 是致命
        if 'cue' in new_pocketed:
            return -1e9, 0, 0
        if ('8' in new_pocketed) and (not can_shoot_8):
            return -1e9, 0, 0

        is_legal, _ = self._check_first_contact(shot, valid_targets)
        if not is_legal:
            return -1e9, 0, 0

        own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in valid_targets and bid not in ['cue', '8']]

        # 进攻导向：更偏向“清球速度”，减少因 60 杆上限导致的点数输。
        base = analyze_shot_for_reward(shot, {bid: copy.deepcopy(ball) for bid, ball in balls.items()}, valid_targets)
        progress = 80.0 * len(own_pocketed) - 30.0 * len(enemy_pocketed)

        # 若没进球，适度惩罚（鼓励尽量争取继续出杆）
        no_pocket_penalty = -12.0 if len(new_pocketed) == 0 else 0.0

        return float(base + progress + no_pocket_penalty), len(own_pocketed), len(enemy_pocketed)

    def _geometric_shot(self, cue_pos, target_pos, pocket_pos):
        """几何预瞄"""
        ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
        cue_to_ghost = ghost_pos - np.array(cue_pos[:2])
        phi = self._angle_to_phi(self._normalize(cue_to_ghost))
        dist = self._distance(cue_pos, ghost_pos)
        V0 = np.clip(1.8 + dist * 2.0, 1.5, 7.0)  # 降低上限7.5→7.0
        return {'V0': float(V0), 'phi': float(phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0}

    def _defense_shot(self, balls, my_targets, table):
        """防守模式"""
        cue_pos = balls['cue'].state.rvw[0]
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        
        if not valid_targets:
            valid_targets = ['8']
        
        # 选择“遮挡最少”的合法目标，避免防守球也打出首球犯规
        candidates = []
        for tid in valid_targets:
            if tid not in balls or balls[tid].state.s == 4:
                continue
            tpos = balls[tid].state.rvw[0]
            ob = self._count_obstructions(balls, cue_pos, tpos, exclude_ids=['cue', tid])
            dist = self._distance(cue_pos, tpos)
            candidates.append((ob, dist, tid))
        candidates.sort()

        # 尝试若干“直接合法首碰 + 保证一定力度”的防守击球，并用快速安全测试过滤
        valid_targets_now, _ = self._get_valid_targets(balls, my_targets)
        phi_offsets = [0.0, 1.0, -1.0, 2.0, -2.0]
        for _, dist, tid in candidates[:6]:
            base_phi = self._angle_to_phi(self._normalize(balls[tid].state.rvw[0] - cue_pos))
            # 防守不应太慢：太慢更容易出现“无进球且未碰库”的回滚犯规
            base_v = float(np.clip(2.2 + dist * 0.6, 2.2, 5.0))
            for dphi in phi_offsets:
                action = {
                    'V0': base_v,
                    'phi': float((base_phi + dphi) % 360),
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0,
                }
                # 防守也可能触发白球进袋/误打黑8/首球犯规，安全阈值不要太低
                if self._is_action_safe(action, balls, table=table, valid_targets=valid_targets_now, simulations=8):
                    return action

        # 如果找不到安全防守球：改为严格寻找“任意安全动作”，避免直接返回未验证动作
        any_safe = self._find_any_safe_action(balls, table, my_targets, attempts=50, safety_sims=6)
        if any_safe is not None:
            return any_safe

        return self._random_action()

    # ========== 主决策 ==========
    def decision(self, balls, my_targets, table):
        """主决策函数"""
        try:
            # 开球检测
            balls_on_table = [b for k, b in balls.items() 
                            if k != 'cue' and b.state.s != 4]
            if len(balls_on_table) == 15:
                # 仅当球型处于“紧密球堆(三角架)”时才视为开球。
                # 否则（例如开球后无人进球，仍有15球在台面但已散开）继续走正常策略。
                try:
                    pos = np.asarray([b.state.rvw[0][:2] for b in balls_on_table], dtype=float)
                    center = pos.mean(axis=0)
                    mean_r = float(np.mean(np.linalg.norm(pos - center, axis=1)))
                except Exception:
                    mean_r = 999.0

                if mean_r < 0.12:
                    if self.debug_mode:
                        print("   [NewAgent] 🎱 开球")
                    action = self.get_break_shot_for_targets(balls, my_targets, table)
                    return self._finalize_action(action, balls, table, my_targets, safety_sims=12)
            
            # 选择目标
            candidates = self._choose_top_targets(balls, my_targets, table, top_k=10)
            if not candidates:
                action = self._defense_shot(balls, my_targets, table)
                return self._finalize_action(action, balls, table, my_targets, safety_sims=8)

            # 多候选策略：
            # - 前2个候选做一次贝叶斯微调（成本较高但更有希望“稳进”）
            # - 其余候选直接走几何+验证+finalize（防止只盯一个路线卡死）
            best_action = None
            best_q = -1e18

            for idx, (tid, pid, _) in enumerate(candidates):
                cue_pos = balls['cue'].state.rvw[0]
                target_pos = balls[tid].state.rvw[0]
                pocket_pos = table.pockets[pid].center

                if idx == 0:
                    if self.debug_mode:
                        print(f"   [NewAgent] 🎯 目标候选: {tid} → 袋口: {pid} (top-{len(candidates)})")
                elif self.debug_mode:
                    print(f"   [DEBUG] 候选{idx+1}: {tid}->{pid}")

                geo_action = self._geometric_shot(cue_pos, target_pos, pocket_pos)

                if idx < 2:
                    cand = self._optimized_search(geo_action, balls, my_targets, table)
                else:
                    cand = geo_action

                cand = self._validate_and_adjust(cand, balls, table, my_targets)
                cand = self._finalize_action(cand, balls, table, my_targets, safety_sims=10)

                q, own_cnt, _ = self._quick_shot_quality(cand, balls, table, my_targets)
                if q > best_q:
                    best_q = q
                    best_action = cand

                # 若找到“确定性可进己方球”的方案，优先立即采用
                if own_cnt > 0:
                    return best_action

            if best_action is not None:
                return best_action

            action = self._defense_shot(balls, my_targets, table)
            return self._finalize_action(action, balls, table, my_targets, safety_sims=8)
        
        except Exception as e:
            print(f"   [ERROR] 决策失败: {e}")
            import traceback
            traceback.print_exc()

            # 不再返回随机动作（会显著抬高 no_hit / cue_pocket / first_foul）
            try:
                safe = self._find_any_safe_action(balls, table, my_targets, attempts=60, safety_sims=6)
                if safe is not None:
                    return safe
                fallback = self._defense_shot(balls, my_targets, table)
                return self._finalize_action(fallback, balls, table, my_targets, safety_sims=8)
            except Exception:
                # 最终兜底：绝不返回随机/跳球（会显著抬高首球犯规/误打黑8）
                try:
                    valid_targets, _ = self._get_valid_targets(balls, my_targets)
                    cue_pos = balls['cue'].state.rvw[0]
                    target_ids = [tid for tid in valid_targets if tid in balls and balls[tid].state.s != 4]
                    if target_ids:
                        target_ids.sort(key=lambda tid: self._distance(cue_pos, balls[tid].state.rvw[0]))
                        tid = target_ids[0]
                        base_phi = self._angle_to_phi(self._normalize(balls[tid].state.rvw[0] - cue_pos))
                        return {
                            'V0': 3.2,
                            'phi': float(base_phi),
                            'theta': 0.0,
                            'a': 0.0,
                            'b': 0.0,
                        }
                except Exception:
                    pass
                return {
                    'V0': 3.0,
                    'phi': 0.0,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0,
                }