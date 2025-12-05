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
                    try:
                        pt.simulate(shot, inplace=True, max_events=200)
                    except Exception:
                        return -500
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
    Phase 10: The Robust Dominator (稳健统治者)

    核心突破：
    1. 抗噪测试 (Robustness Check): 引入蒙特卡洛模拟，对候选动作添加环境噪声进行多次验证。
       只有在噪声下依然稳定的进球路线才会被采纳，彻底消除“莫名其妙打丢”的失误。
    2. 动态风险评估: 宁可打进率 100% 的简单球，也不打进率 50% 的神仙球。
    3. 继承 Phase 9 的暴力开球与防守逻辑。
    """

    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        # 必须与环境噪声保持一致，用于自我测试
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        print("NewAgent (Phase 10) 已初始化 - 稳健统治模式")

    def _calculate_angle_degrees(self, v):
        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle < 0: angle += 360
        return angle

    def get_aim_info(self, target_ball, pocket, cue_ball):
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

    def check_next_shot_exist(self, balls, my_targets, table):
        """简单的下球路线检查"""
        cue_ball = balls['cue']
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        targets = remaining if remaining else ['8']
        for tid in targets:
            if balls[tid].state.s == 4: continue
            for pid, pocket in table.pockets.items():
                _, cut_angle, _ = self.get_aim_info(balls[tid], pocket, cue_ball)
                if cut_angle < 75: return True
        return False

    def get_break_shot(self, balls):
        """Phase 9 的完美开球"""
        target = balls['1']
        cue = balls['cue']
        vec = target.state.rvw[0] - cue.state.rvw[0]
        phi = self._calculate_angle_degrees(vec)
        return {'V0': 8.0, 'phi': phi, 'theta': 0, 'a': 0.01, 'b': -0.08}

    def simulate_with_noise(self, shot_params, table, balls, n_sims=3):
        """
        抗噪测试核心函数
        对同一个动作进行 n_sims 次带噪声的模拟，返回成功进球的次数和平均分
        """
        success_count = 0
        total_score = 0
        min_score = 9999.0

        sim_table = copy.deepcopy(table)

        for _ in range(n_sims):
            # 添加噪声
            noisy_action = {
                'V0': shot_params['V0'] + np.random.normal(0, self.noise_std['V0']),
                'phi': shot_params['phi'] + np.random.normal(0, self.noise_std['phi']),
                'theta': 0,
                'a': shot_params.get('a', 0) + np.random.normal(0, self.noise_std['a']),
                'b': shot_params.get('b', 0) + np.random.normal(0, self.noise_std['b'])
            }

            # 限制范围
            noisy_action['V0'] = np.clip(noisy_action['V0'], 0.1, 8.0)

            # 模拟
            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**noisy_action)

            # 必须加 max_events=200 防止死锁
            try:
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                # 如果模拟卡死，直接判为极差
                return 0, -5000, -5000

            # 评分 (简化的单次评分)
            score = 0
            new_pocketed = [bid for bid, b in sim_balls.items() if b.state.s == 4 and balls[bid].state.s != 4]
            cue_potted = 'cue' in new_pocketed
            eight_potted = '8' in new_pocketed
            target_potted = shot_params['target'] in new_pocketed

            # 生死判定
            is_shooting_8 = (shot_params['target'] == '8')

            if eight_potted:
                if not is_shooting_8 or cue_potted:
                    score = -5000;  # 判负
                else:
                    score = 5000;  # 赢了
            elif cue_potted:
                score = -2000  # 洗袋
            elif target_potted:
                score = 100
                score -= shot_params['cut'] * 0.2
            else:
                score = -50
                # 没进球时的防守检查略过，主要看能不能进

            total_score += score
            if score < min_score: min_score = score

            # 统计成功进球次数 (不算黑8判负的情况)
            if target_potted and not cue_potted and not (eight_potted and not is_shooting_8):
                success_count += 1

        return success_count, total_score / n_sims, min_score

    def decision(self, balls, my_targets, table):
        try:
            cue_ball = balls['cue']

            # 0. 开球
            balls_on_table = [b for k, b in balls.items() if k != 'cue' and b.state.s != 4]
            if len(balls_on_table) == 15:
                print("[Robust] 🎱 完美暴力开球")
                return self.get_break_shot(balls)

            remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_8 = len(remaining_targets) == 0
            targets_to_search = remaining_targets if not is_shooting_8 else ['8']

            # 1. 进攻海选
            candidates = []
            for tid in targets_to_search:
                if balls[tid].state.s == 4: continue
                for pid, pocket in table.pockets.items():
                    aim_phi, cut_angle, dist = self.get_aim_info(balls[tid], pocket, cue_ball)
                    if cut_angle > 82: continue

                    # 生成候选: 标准力 & 小力
                    v_base = np.clip(2.0 + dist * 2.3, 2.0, 7.5)
                    # 优先考虑中等力度，最稳
                    candidates.append({'target': tid, 'phi': aim_phi, 'cut': cut_angle, 'V0': v_base})
                    if dist < 1.0:
                        candidates.append(
                            {'target': tid, 'phi': aim_phi, 'cut': cut_angle, 'V0': np.clip(v_base * 0.7, 1.5, 4.0)})
                    # 大力修正 (针对切球)
                    if cut_angle < 50:
                        candidates.append(
                            {'target': tid, 'phi': aim_phi, 'cut': cut_angle, 'V0': np.clip(v_base * 1.4, 3.0, 8.0)})

            candidates.sort(key=lambda x: x['cut'])
            top_candidates = candidates[:6]  # 只验证前6个

            best_action = None
            best_robust_score = -99999.0

            # 2. 抗噪模拟 (Robustness Check)
            # 对每个候选进行 3 次带噪声模拟
            for cand in top_candidates:
                # n_sims=3: 模拟3次。必须至少进2次才考虑，进3次最好。
                success_count, avg_score, min_score = self.simulate_with_noise(cand, table, balls, n_sims=3)

                # 过滤高风险球：
                # 如果3次里有1次洗袋或判负(min_score < -1000)，绝对不打
                if min_score < -1000: continue

                # 稳定性评分：
                # 成功率权重极高。成功3次 > 成功2次 >> 成功1次
                robust_score = success_count * 1000 + avg_score

                # 走位加分 (仅对稳进的球计算走位)
                if success_count >= 2 and not is_shooting_8:
                    # 快速检查一次无噪声的走位
                    # (为了节省时间，这里不再带噪声模拟走位，只基于无噪声几何检查)
                    # 这里简化处理：直接用 avg_score 里的距离/切角因子
                    pass

                if robust_score > best_robust_score:
                    best_robust_score = robust_score
                    best_action = cand
                    # 记录该动作的成功率，用于日志
                    best_action['success_rate'] = success_count

            # 3. 决策阈值
            # 如果最佳球的成功率 < 2/3 (即3次只进不到了2次)，说明很不稳，不如防守
            if best_action and best_action['success_rate'] >= 2:
                print(f"[Robust] 🎯 稳健进攻: {best_action['target']} (稳度:{best_action['success_rate']}/3)")
                return {'V0': best_action['V0'], 'phi': best_action['phi'], 'theta': 0, 'a': 0, 'b': 0}

            # 4. 顶级防守 (Elite Safety)
            print("[Robust] 🛡️ 进攻风险大，执行防守")
            # 找最近的球，尝试踢开
            safety_candidates = []
            for tid in targets_to_search:
                if balls[tid].state.s == 4: continue
                dist = np.linalg.norm(balls[tid].state.rvw[0] - cue_ball.state.rvw[0])
                if dist > 1.2: continue  # 太远不碰

                vec = balls[tid].state.rvw[0] - cue_ball.state.rvw[0]
                phi = self._calculate_angle_degrees(vec)
                safety_candidates.append({'V0': 3.0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0})
                safety_candidates.append({'V0': 2.0, 'phi': phi + 2, 'theta': 0, 'a': 0, 'b': 0})
                safety_candidates.append({'V0': 2.0, 'phi': phi - 2, 'theta': 0, 'a': 0, 'b': 0})

            # 简单的防守选择：选那个肯定不洗袋的
            for shot in safety_candidates:
                # 快速单次验证
                success, avg, min_s = self.simulate_with_noise(dict(target='none', cut=0, **shot), table, balls,
                                                               n_sims=1)
                if min_s > -500:  # 安全
                    return shot

            return self._random_action()

        except Exception as e:
            print(f"Error: {e}")
            return self._random_action()