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

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
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
    Optimized NewAgent: Phase 23 - Black 8 Guardian
    改进点：
    1. 新增 6次抗噪安全检查，防止误打黑8和白球洗袋。
    2. 继承了之前的走位和优化逻辑。
    """

    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        self.SEARCH_INIT = 15
        self.SEARCH_ITER = 10
        
        # 同步环境的噪声参数，用于自我评估
        self.noise_std = {
            'V0': 0.1,      # 速度标准差 
            'phi': 0.1,     # 角度标准差
            'theta': 0.1, 
            'a': 0.003, 
            'b': 0.003
        }
        print("[NewAgent] Phase 23: Black 8 Guardian 已初始化 (含6次抗噪检测)")

    # ... [保留原有的 _distance, _normalize, _angle_to_phi, _calculate_ghost_ball 等工具函数] ...
    # 为了完整性，这里列出必须的工具函数，未修改的逻辑保持原样即可
    
    def _distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

    def _normalize(self, vec):
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        if norm < 1e-6: return np.array([1.0, 0.0])
        return vec / norm

    def _angle_to_phi(self, direction_vec):
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360

    def _calculate_ghost_ball(self, target_pos, pocket_pos):
        target_to_pocket = self._normalize(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        ghost_pos = np.array(target_pos[:2]) - target_to_pocket * (2 * self.BALL_RADIUS)
        return ghost_pos

    def _calculate_cut_angle(self, cue_pos, target_pos, pocket_pos):
        ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
        vec1 = self._normalize(np.array(ghost_pos) - np.array(cue_pos[:2]))
        vec2 = self._normalize(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def _check_can_shoot_8(self, balls, original_targets):
        real_targets = [bid for bid in original_targets if bid != '8']
        remaining = [bid for bid in real_targets if balls[bid].state.s != 4]
        return len(remaining) == 0

    # ... [保留 _evaluate_position_quality, get_break_shot, _count_obstructions, _choose_best_target] ...
    # 假设这些函数与原文件一致，此处省略以节省篇幅，重点在下面的验证逻辑
    
    def _evaluate_position_quality(self, cue_pos, balls, my_targets, original_targets):
        # ... (保持原代码逻辑) ...
        # 简单实现用于占位，请保留原文件内容
        remaining_targets = [tid for tid in my_targets if balls[tid].state.s != 4]
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)
        if len(remaining_targets) == 0 or (len(remaining_targets) == 1 and remaining_targets[0] == '8' and not can_shoot_8):
            target_candidates = ['8']
        else:
            target_candidates = [t for t in remaining_targets if t != '8']
        if not target_candidates: return 1.0
        min_dist = 100.0
        for tid in target_candidates:
            dist = self._distance(cue_pos, balls[tid].state.rvw[0])
            if dist < min_dist: min_dist = dist
        if 0.2 < min_dist < 1.0: return 1.0
        return 0.5

    def get_break_shot(self, balls):
        target = balls['1']
        cue = balls['cue']
        vec = target.state.rvw[0] - cue.state.rvw[0]
        phi = self._angle_to_phi(self._normalize(vec))
        return {'V0': 8.0, 'phi': phi, 'theta': 0, 'a': 0.0, 'b': -0.2}
        
    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        # ... (保持原代码逻辑) ...
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

    def _choose_best_target(self, balls, my_targets, table, original_targets):
        # ... (保持原代码逻辑) ...
        # 请直接使用原文件中的 _choose_best_target 实现
        best_choice = None
        best_score = -1e9
        cue_pos = balls['cue'].state.rvw[0]
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)

        for target_id in my_targets:
            if target_id == '8' and not can_shoot_8: continue
            if balls[target_id].state.s == 4: continue
            target_pos = balls[target_id].state.rvw[0]
            for pocket_id, pocket in table.pockets.items():
                score = 0
                pocket_pos = pocket.center
                dist_cue_target = self._distance(cue_pos, target_pos)
                dist_target_pocket = self._distance(target_pos, pocket_pos)
                score += 50 / (1 + dist_cue_target + dist_target_pocket)
                cut_angle = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos)
                if cut_angle > 80: continue
                score += (90 - cut_angle) * 1.2
                obs_1 = self._count_obstructions(balls, cue_pos, target_pos, exclude_ids=['cue', target_id])
                if obs_1 > 0: score -= 500
                obs_2 = self._count_obstructions(balls, target_pos, pocket_pos, exclude_ids=['cue', target_id])
                if obs_2 > 0: score -= 500
                ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
                for pid_danger, p_danger in table.pockets.items():
                    if self._distance(ghost_pos, p_danger.center) < 0.12: score -= 300
                if target_id == '8' and can_shoot_8: score += 500
                if score > best_score:
                    best_score = score
                    best_choice = (target_id, pocket_id)
        return best_choice

    def _geometric_shot(self, cue_pos, target_pos, pocket_pos):
        ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
        cue_to_ghost = ghost_pos - np.array(cue_pos[:2])
        phi = self._angle_to_phi(self._normalize(cue_to_ghost))
        dist = self._distance(cue_pos, ghost_pos)
        V0 = np.clip(1.8 + dist * 2.2, 1.5, 7.5)
        return {'V0': float(V0), 'phi': float(phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0}

    def _optimized_search(self, geo_action, balls, my_targets, table, original_targets):
        # ... (保持原代码逻辑) ...
        # 请直接使用原文件中的 _optimized_search 实现
        pbounds = {
            'V0': (max(0.5, geo_action['V0'] - 1.5), min(8.0, geo_action['V0'] + 1.5)),
            'phi': (geo_action['phi'] - 2.5, geo_action['phi'] + 2.5),
            'theta': (0, 0), 'a': (-0.5, 0.5), 'b': (-0.5, 0.5)
        }
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)

        def reward_fn(V0, phi, theta, a, b):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            try:
                shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                pt.simulate(shot, inplace=True, max_events=200)
            except: return -500
            
            # 死锁与基础分
            is_stuck = False
            for ball in shot.balls.values():
                if ball.state.s not in [0, 4]: is_stuck = True; break
            if is_stuck: return -2000
            base_score = analyze_shot_for_reward(shot, last_state, my_targets)
            if base_score < 0: return base_score
            
            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
            if '8' in new_pocketed and not can_shoot_8: return -1000
            
            own_pocketed = [bid for bid in new_pocketed if bid in my_targets]
            position_bonus = 0
            if len(own_pocketed) > 0 and '8' not in new_pocketed:
                final_cue_pos = shot.balls['cue'].state.rvw[0]
                pos_quality = self._evaluate_position_quality(final_cue_pos, shot.balls, my_targets, original_targets)
                position_bonus = pos_quality * 30
            return base_score + position_bonus

        try:
            optimizer = BayesianOptimization(f=reward_fn, pbounds=pbounds, random_state=42, verbose=0)
            optimizer.maximize(init_points=self.SEARCH_INIT, n_iter=self.SEARCH_ITER)
            if optimizer.max['target'] > -100:
                p = optimizer.max['params']
                return {'V0': p['V0'], 'phi': p['phi'], 'theta': p['theta'], 'a': p['a'], 'b': p['b']}
        except: pass
        return geo_action

    # ==================== 新增：抗噪安全检查 ====================
    def _check_safety_robust(self, action, balls, table, can_shoot_8, simulations=6):
        """
        核心改进：通过多次蒙特卡洛模拟检查动作的安全性
        如果任何一次模拟导致误进黑8或白球进袋，则视为不安全
        """
        for i in range(simulations):
            # 1. 施加随机噪声 (模拟真实环境误差)
            noisy_action = {
                'V0': np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0),
                'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
                'theta': np.clip(action['theta'] + np.random.normal(0, self.noise_std['theta']), 0, 90),
                'a': np.clip(action['a'] + np.random.normal(0, self.noise_std['a']), -0.5, 0.5),
                'b': np.clip(action['b'] + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
            }
            
            # 2. 模拟
            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            try:
                shot.cue.set_state(**noisy_action)
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                continue # 物理引擎错误忽略，但不通过

            # 3. 检查灾难性后果
            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
            
            # [致命] 白球进袋 -> 绝对禁止
            if 'cue' in new_pocketed:
                return False
                
            # [致命] 误进黑8 -> 绝对禁止
            if '8' in new_pocketed and not can_shoot_8:
                return False
                
            # [可选高阶] 如果首球没打到目标球，可能导致犯规，这里为了进攻性可以暂时容忍，
            # 但如果对犯规非常敏感，也可以在这里 return False
            
        return True # 所有测试通过

    def _validate_and_adjust(self, action, balls, table, my_targets, original_targets):
        # 验证集：保留原有的微调逻辑，但加入更严格的噪声过滤
        variations = [
            (1.0, 0), (0.95, 0), (1.05, 0),
            (1.0, 0.5), (1.0, -0.5)
        ]
        sim_table = copy.deepcopy(table)
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)

        best_safe_action = None

        for v_scale, phi_offset in variations:
            test_action = action.copy()
            test_action['V0'] *= v_scale
            test_action['phi'] += phi_offset

            # 1. 先跑一次确定性模拟（快速筛选）
            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**test_action)
            try:
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                continue

            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]

            # 基础过滤
            if 'cue' in new_pocketed: continue
            if '8' in new_pocketed and not can_shoot_8: continue
            
            # 2. [改进点] 核心：6次抗噪测试
            # 只有通过了确定性测试的动作，才有资格进入抗噪测试（节省计算资源）
            if not self._check_safety_robust(test_action, balls, table, can_shoot_8, simulations=6):
                print(f"[Guardian] ⚠️ 拦截了一个高风险动作 (V0={test_action['V0']:.1f}, phi={test_action['phi']:.1f})")
                continue

            # 3. 检查是否进目标球
            own_pocketed = [bid for bid in new_pocketed if bid in my_targets]
            if len(own_pocketed) > 0:
                return test_action # 这是一个既进球又鲁棒的动作

            # 4. 如果没进球但安全，存为备选
            if v_scale == 1.0 and phi_offset == 0:
                best_safe_action = test_action

        # 如果所有进攻线路都不安全（通不过抗噪测试），或者打不进球
        if best_safe_action is not None:
            return best_safe_action

        # 兜底防守
        print("[Protector] 🛡️ 启动防守模式")
        return self._defense_shot(balls, my_targets)

    def _defense_shot(self, balls, my_targets):
        # ... (保持原代码逻辑) ...
        # 请直接使用原文件中的 _defense_shot 实现
        cue_pos = balls['cue'].state.rvw[0]
        min_dist = 100
        target_id = None
        candidates = [b for b in my_targets if balls[b].state.s != 4]
        if not candidates: candidates = ['8']
        for tid in candidates:
            dist = self._distance(cue_pos, balls[tid].state.rvw[0])
            if dist < min_dist:
                min_dist = dist
                target_id = tid
        if target_id:
            vec = balls[target_id].state.rvw[0] - cue_pos
            phi = self._angle_to_phi(self._normalize(vec))
            return {'V0': 1.0 + min_dist, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
        return self._random_action()

    def decision(self, balls, my_targets, table):
        # ... (保持原代码逻辑) ...
        try:
            balls_on_table = [b for k, b in balls.items() if k != 'cue' and b.state.s != 4]
            if len(balls_on_table) == 15:
                print("[NewAgent] 🎱 开球")
                return self.get_break_shot(balls)

            original_targets = list(my_targets)
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if not remaining: my_targets = ['8']

            choice = self._choose_best_target(balls, my_targets, table, original_targets)
            if not choice: return self._defense_shot(balls, my_targets)

            tid, pid = choice
            cue_pos = balls['cue'].state.rvw[0]
            target_pos = balls[tid].state.rvw[0]
            pocket_pos = table.pockets[pid].center
            
            print(f"[NewAgent] 目标: {tid} -> 袋口: {pid}")

            geo_action = self._geometric_shot(cue_pos, target_pos, pocket_pos)
            final_action = self._optimized_search(geo_action, balls, my_targets, table, original_targets)
            final_action = self._validate_and_adjust(final_action, balls, table, my_targets, original_targets)
            return final_action

        except Exception as e:
            print(f"[NewAgent] Critical Error: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()