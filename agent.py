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
    Phase 5: Grandmaster (深度规划版)
    核心能力：
    1. 多力度尝试：对同一线路尝试不同力度，寻找最佳走位。
    2. 绝境避免：进球后检查是否被斯诺克，拒绝死路。
    3. 智能防守：无球可进时，执行必得的安全球，拒绝送分。
    """

    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        print("NewAgent (Grandmaster) 已初始化 - 冠军模式")

    def _calculate_angle_degrees(self, v):
        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle < 0: angle += 360
        return angle

    def get_aim_info(self, target_ball, pocket, cue_ball):
        # --- 几何计算基础 ---
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

    def check_next_shot_availability(self, balls, my_targets, table):
        """
        快速几何检查：当前局面下，是否至少有一颗球是好打的？
        用于判断走位是否成功。
        """
        cue_ball = balls['cue']
        # 如果打完了，下一个目标是黑8
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        targets = remaining if remaining else ['8']

        has_good_shot = False

        for tid in targets:
            if balls[tid].state.s == 4: continue
            for pid, pocket in table.pockets.items():
                _, cut_angle, _ = self.get_aim_info(balls[tid], pocket, cue_ball)
                # 只要有一颗球的切角 < 70度，就认为活着
                if cut_angle < 70:
                    return True  # 只要有一条活路就行
        return False

    def decision(self, balls, my_targets, table):
        try:
            remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_8 = len(remaining_targets) == 0
            targets_to_search = remaining_targets if not is_shooting_8 else ['8']
            cue_ball = balls['cue']

            # 1. 几何海选 (生成候选动作)
            candidates = []

            # 简单的防守备选：记录离母球最近的球，万一没球打，就轻轻摸它一下
            safety_target = None
            min_dist_safety = 999.0

            for tid in targets_to_search:
                if balls[tid].state.s == 4: continue

                # 记录防守信息
                dist_to_ball = np.linalg.norm(balls[tid].state.rvw[0] - cue_ball.state.rvw[0])
                if dist_to_ball < min_dist_safety:
                    min_dist_safety = dist_to_ball
                    vec_safety = balls[tid].state.rvw[0] - cue_ball.state.rvw[0]
                    safety_target = {
                        'phi': self._calculate_angle_degrees(vec_safety),
                        'V0': 0.5 + dist_to_ball * 1.0  # 极轻力度
                    }

                for pid, pocket in table.pockets.items():
                    aim_phi, cut_angle, dist = self.get_aim_info(balls[tid], pocket, cue_ball)
                    if cut_angle > 82: continue

                    # === 策略升级：一球多策 ===
                    # 针对同一个角度，生成 2-3 种力度的候选
                    # 1. 标准力度 (刚好够进球 + 一点余量)
                    v_normal = np.clip(2.0 + dist * 2.3, 2.0, 7.5)
                    candidates.append(
                        {'target': tid, 'phi': aim_phi, 'cut': cut_angle, 'V0': v_normal, 'type': 'normal'})

                    # 2. 大力出奇迹 (仅当切角不大时，大力可以减少静摩擦偏差，且容易炸散球堆)
                    if cut_angle < 45 and dist < 1.5:
                        v_hard = np.clip(v_normal * 1.4, 3.0, 8.0)
                        candidates.append(
                            {'target': tid, 'phi': aim_phi, 'cut': cut_angle, 'V0': v_hard, 'type': 'hard'})

                    # 3. 温柔一推 (仅当距离近时，为了精准走位)
                    if dist < 0.8:
                        v_soft = np.clip(v_normal * 0.7, 1.5, 4.0)
                        candidates.append(
                            {'target': tid, 'phi': aim_phi, 'cut': cut_angle, 'V0': v_soft, 'type': 'soft'})

            # 排序：只验证最有希望的 6 个方案 (包含不同力度的变种)
            candidates.sort(key=lambda x: x['cut'])
            top_candidates = candidates[:6]

            best_action = None
            best_score = -99999.0

            # 2. 物理模拟验证
            sim_table = copy.deepcopy(table)

            for cand in top_candidates:
                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                shot.cue.set_state(V0=cand['V0'], phi=cand['phi'], theta=0, a=0, b=0)

                pt.simulate(shot, inplace=True, max_events=200)

                # --- 评分系统 v3.0 ---
                score = 0

                new_pocketed = [bid for bid, b in sim_balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
                cue_potted = 'cue' in new_pocketed
                eight_potted = '8' in new_pocketed
                target_potted = cand['target'] in new_pocketed

                # A. 生死判定 (Death Checks)
                if eight_potted:
                    if not is_shooting_8 or cue_potted:
                        score = -100000; continue  # 判负，直接跳过
                    else:
                        score = 100000; break  # 赢了！直接选它！
                if cue_potted:
                    score = -5000;
                    continue  # 洗袋，跳过

                # B. 进球逻辑
                if target_potted:
                    score += 100
                    score -= cand['cut'] * 0.2  # 稍微惩罚大切角

                    # C. 绝境检测 (Next-Shot Guarantee)
                    # 如果这杆打完，不是黑8，且还没赢
                    if not is_shooting_8:
                        # 检查打完后有没有活路
                        has_next = self.check_next_shot_availability(sim_balls, my_targets, sim_table)
                        if has_next:
                            score += 50  # 很好，路是通的
                        else:
                            score -= 80  # 糟糕，打进这球我就被斯诺克了 (这种球不如不打)
                else:
                    # 没进球
                    score -= 50
                    # 检查是否犯规(没碰到球)
                    target_moved = np.linalg.norm(
                        sim_balls[cand['target']].state.rvw[0] - balls[cand['target']].state.rvw[0]) > 0.001
                    if not target_moved: score -= 200

                if score > best_score:
                    best_score = score
                    best_action = cand

            # 3. 最终决策
            if best_action and best_score > -200:
                print(
                    f"[Grandmaster] 🎯 锁定目标: {best_action['target']} (力度:{best_action['type']}), 评分:{best_score:.1f}")
                return {'V0': best_action['V0'], 'phi': best_action['phi'], 'theta': 0, 'a': 0, 'b': 0}

            # 4. 智能防守 (Smart Safety)
            # 如果上面没找到靠谱的进攻机会，千万别 random！
            # 找最近的球，轻碰一下，避免犯规。
            if safety_target:
                print(f"[Grandmaster] 🛡️ 启动防守: 轻推球 {safety_target['V0']:.2f}")
                return {'V0': safety_target['V0'], 'phi': safety_target['phi'], 'theta': 0, 'a': 0, 'b': 0}

            print("[Grandmaster] ⚠️ 绝境，随机防守")
            return self._random_action()

        except Exception as e:
            print(f"Error: {e}")
            return self._random_action()