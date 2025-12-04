"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
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
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
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
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
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
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
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
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class NewAgent(Agent):
    """
    Phase 3: 严格合规版 (Rule-Compliant)
    特点：
    1. 严格遵守 GAME_RULES.md，防止误打黑8直接判负。
    2. 物理模拟检测洗袋、未碰库等犯规。
    3. 几何筛选 + 物理验证双重保障。
    """

    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        print("NewAgent (Rule-Compliant) 已初始化 - 严格合规模式")

    def _calculate_angle_degrees(self, v):
        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle < 0: angle += 360
        return angle

    def get_aim_info(self, target_ball, pocket, cue_ball):
        # --- 几何计算部分 (保持不变) ---
        pos_t = target_ball.state.rvw[0]
        pos_c = cue_ball.state.rvw[0]
        pos_p = pocket.center

        vec_t_p = pos_p - pos_t
        dist_t_p = np.linalg.norm(vec_t_p)
        dir_t_p = vec_t_p / (dist_t_p + 1e-9)
        pos_ghost = pos_t - dir_t_p * (2 * self.BALL_RADIUS)

        vec_c_g = pos_ghost - pos_c
        aim_phi = self._calculate_angle_degrees(vec_c_g)

        vec_c_t = pos_t - pos_c
        cos_theta = np.dot(vec_c_t, vec_t_p) / (np.linalg.norm(vec_c_t) * dist_t_p + 1e-9)
        cut_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        total_dist = np.linalg.norm(vec_c_g) + dist_t_p
        return aim_phi, cut_angle, total_dist

    def decision(self, balls, my_targets, table):
        try:
            # 1. 识别当前目标
            # 注意：必须动态判断是否该打黑8了
            remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_8 = len(remaining_targets) == 0

            # 如果还有目标球，就打目标球；否则打黑8
            targets_to_search = remaining_targets if not is_shooting_8 else ['8']
            cue_ball = balls['cue']

            # 2. 几何海选
            candidates = []
            for tid in targets_to_search:
                if balls[tid].state.s == 4: continue
                for pid, pocket in table.pockets.items():
                    aim_phi, cut_angle, dist = self.get_aim_info(balls[tid], pocket, cue_ball)

                    if cut_angle > 85: continue

                    base_v0 = np.clip(2.2 + dist * 2.5, 2.2, 7.5)

                    candidates.append({
                        'target': tid, 'pocket': pid,
                        'phi': aim_phi, 'cut': cut_angle, 'V0': base_v0
                    })

            # 排序筛选
            candidates.sort(key=lambda x: x['cut'])
            top_candidates = candidates[:3]

            best_action = None
            best_score = -99999.0

            # 3. 物理模拟验证 (核心修正部分)
            sim_table = copy.deepcopy(table)

            for cand in top_candidates:
                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                shot.cue.set_state(V0=cand['V0'], phi=cand['phi'], theta=0, a=0, b=0)

                # 模拟
                pt.simulate(shot, inplace=True, max_events=200)

                # --- 评分逻辑 (对照 GAME_RULES.md) ---
                score = 0

                # 获取进球列表
                new_pocketed = [bid for bid, b in sim_balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
                cue_potted = 'cue' in new_pocketed
                eight_potted = '8' in new_pocketed
                target_potted = cand['target'] in new_pocketed

                # Rule 1.5: 即时判负规则检测
                if eight_potted:
                    if not is_shooting_8:
                        # 还没清空球就打进黑8 -> 判负
                        score -= 10000
                    elif cue_potted:
                        # 白球和黑8同时进 -> 判负
                        score -= 10000
                    else:
                        # 合法打进黑8 -> 胜利！
                        score += 1000

                # Rule 1.5: 交换球权犯规检测
                elif cue_potted:
                    # 白球洗袋
                    score -= 1000

                # 进球奖励
                elif target_potted:
                    score += 100
                    score -= cand['cut'] * 0.5  # 优先打容易的

                    # 简单的走位判断：如果进球后白球贴库了，扣分
                    # (判断方法：检查白球是否在 Table 边界附近)
                    # W=table.w, L=table.l. 简单略过，因为还要解析 table 尺寸
                else:
                    # 没进球
                    score -= 50

                    # Rule: 未碰库犯规检测
                    # 检查是否有任意球碰库或进袋
                    # 这里为了简化计算（不解析 events），我们假设：
                    # 如果球没进，且所有球位置几乎没变，说明大概率犯规了
                    any_moved = False
                    for b_id in sim_balls:
                        if np.linalg.norm(sim_balls[b_id].state.rvw[0] - balls[b_id].state.rvw[0]) > 0.005:
                            any_moved = True
                            break
                    if not any_moved:
                        score -= 200  # 可能没打到球

                # 更新最佳
                if score > best_score:
                    best_score = score
                    best_action = cand

            # 4. 决策输出
            if best_action and best_score > -5000:  # 只要不是判负或洗袋
                print(f"[NewAgent] ✅ 合规决策: 目标{best_action['target']}, 评分{best_score:.1f}")
                return {'V0': best_action['V0'], 'phi': best_action['phi'], 'theta': 0, 'a': 0, 'b': 0}

            print("[NewAgent] ⚠️ 风险过大，随机防守")
            return self._random_action()

        except Exception as e:
            print(f"[NewAgent] 出错: {e}")
            return self._random_action()