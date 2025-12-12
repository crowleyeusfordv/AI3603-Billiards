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
            
        if len(remaining_own_before) > 0:
            if first_contact_ball_id in opponent_plus_eight:
                foul_first_hit = True
        else:
            if first_contact_ball_id != '8':
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
                    # [修复]: 增加 max_events 限制，防止死锁
                    pt.simulate(shot, inplace=True, max_events=200)
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
    Phase 21: The Absolute Protector (绝对守护者)
    核心修复：
    1. **修复 my_targets 判断逻辑**：使用原始目标球列表判断，而不是动态更新后的
    2. **reward 函数黑8惩罚增强**：在评分函数层面拦截误打黑8
    3. **三重保护机制**：选球阶段 + 优化阶段 + 验证阶段
    """

    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        self.LIGHT_SEARCH_INIT = 5
        self.LIGHT_SEARCH_ITER = 5
        print("[NewAgent] Phase 21: 绝对守护者 已初始化")

    # ==================== 工具函数 ====================
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
        """
        检查是否可以合法打黑8
        关键：使用原始目标球列表（不含'8'）
        """
        # original_targets 应该是 ['1','2',...,'7'] 或 ['9','10',...,'15']
        # 排除 '8' 后检查
        real_targets = [bid for bid in original_targets if bid != '8']
        remaining = [bid for bid in real_targets if balls[bid].state.s != 4]
        return len(remaining) == 0

    # ==================== Layer 0: 开球 ====================
    def get_break_shot(self, balls):
        target = balls['1']
        cue = balls['cue']
        vec = target.state.rvw[0] - cue.state.rvw[0]
        phi = self._angle_to_phi(self._normalize(vec))
        return {'V0': 7.0, 'phi': phi, 'theta': 0, 'a': 0.0, 'b': 0.1}

    # ==================== Layer 1: 目标选择（第一重保护） ====================
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

    def _choose_best_target(self, balls, my_targets, table, original_targets):
        """
        改进：传入原始目标球列表进行判断
        """
        best_choice = None
        best_score = -1e9
        cue_pos = balls['cue'].state.rvw[0]

        # 🔥 修复：使用原始目标球列表判断
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)

        print(f"[Protector] 目标球检查: my_targets={my_targets}, can_shoot_8={can_shoot_8}")

        for target_id in my_targets:
            # 🔥 第一重保护：如果不能打黑8，直接跳过
            if target_id == '8' and not can_shoot_8:
                print(f"[Protector] 🚫 跳过黑8（己方球未清空）")
                continue

            if balls[target_id].state.s == 4:
                continue

            target_pos = balls[target_id].state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                score = 0
                pocket_pos = pocket.center

                dist = self._distance(cue_pos, target_pos)
                score += 50 / (1 + dist)

                cut_angle = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos)
                if cut_angle > 85: continue
                score += (90 - cut_angle) * 0.8

                obs_1 = self._count_obstructions(balls, cue_pos, target_pos, exclude_ids=['cue', target_id])
                score -= obs_1 * 150
                obs_2 = self._count_obstructions(balls, target_pos, pocket_pos, exclude_ids=['cue', target_id])
                score -= obs_2 * 150

                ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
                for pid_danger, p_danger in table.pockets.items():
                    if self._distance(ghost_pos, p_danger.center) < 0.15:
                        score -= 200

                if target_id == '8' and can_shoot_8:
                    score += 300

                if score > best_score:
                    best_score = score
                    best_choice = (target_id, pocket_id)

        return best_choice

    # ==================== Layer 2: 击球生成（第二重保护） ====================
    def _geometric_shot(self, cue_pos, target_pos, pocket_pos):
        ghost_pos = self._calculate_ghost_ball(target_pos, pocket_pos)
        cue_to_ghost = ghost_pos - np.array(cue_pos[:2])
        phi = self._angle_to_phi(self._normalize(cue_to_ghost))
        dist = self._distance(cue_pos, ghost_pos)
        V0 = np.clip(1.8 + dist * 2.0, 1.5, 6.5)
        return {'V0': float(V0), 'phi': float(phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0}

    def _optimized_search(self, geo_action, balls, my_targets, table, original_targets):
        """
        改进：在reward函数中增加黑8误打检测
        """
        pbounds = {
            'V0': (max(1.0, geo_action['V0'] - 1.0), min(7.5, geo_action['V0'] + 1.5)),
            'phi': (geo_action['phi'] - 3, geo_action['phi'] + 3),
            'theta': (0, 0),
            'a': (-0.05, 0.05),
            'b': (-0.05, 0.05)
        }
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 🔥 修复：检查是否可以打黑8
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)

        def reward_fn(V0, phi, theta, a, b):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
            try:
                shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                return -500

            # 🔥 第二重保护：在reward计算中检测黑8误打
            new_pocketed = [bid for bid, b in shot.balls.items()
                            if b.state.s == 4 and last_state[bid].state.s != 4]

            if '8' in new_pocketed and not can_shoot_8:
                print(f"[Protector] ⚠️ 优化器检测到黑8误打，给予极大惩罚")
                return -1000  # 极大惩罚，确保优化器不会选择这类动作

            return analyze_shot_for_reward(shot, last_state, my_targets)

        try:
            optimizer = BayesianOptimization(f=reward_fn, pbounds=pbounds, random_state=1, verbose=0)
            optimizer.maximize(init_points=self.LIGHT_SEARCH_INIT, n_iter=self.LIGHT_SEARCH_ITER)
            if optimizer.max['target'] > 0:
                p = optimizer.max['params']
                return {'V0': p['V0'], 'phi': p['phi'], 'theta': p['theta'], 'a': p['a'], 'b': p['b']}
        except:
            pass
        return geo_action

    # ==================== Layer 3: 验证（第三重保护） ====================
    def _validate_and_adjust(self, action, balls, table, my_targets, original_targets):
        """
        第三重保护：验证阶段再次检查黑8
        """
        variations = [
            (1.0, 0), (0.9, 0), (0.8, 0),
            (0.9, 1), (0.9, -1)
        ]
        sim_table = copy.deepcopy(table)
        safe_action = None

        # 🔥 修复：使用原始目标球判断
        can_shoot_8 = self._check_can_shoot_8(balls, original_targets)

        for v_scale, phi_offset in variations:
            test_action = action.copy()
            test_action['V0'] *= v_scale
            test_action['phi'] += phi_offset

            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(**test_action)

            try:
                pt.simulate(shot, inplace=True, max_events=200)
            except:
                continue

            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]

            if 'cue' in new_pocketed:
                continue

            # 🔥 第三重保护：验证阶段最后防线
            if '8' in new_pocketed and not can_shoot_8:
                print(f"[Protector] 🛡️ 验证阶段拦截黑8误打 (scale={v_scale}, off={phi_offset})")
                continue

            own_pocketed = [bid for bid in new_pocketed if bid in my_targets]
            if len(own_pocketed) > 0:
                print(f"[Protector] ✅ 验证通过 (scale={v_scale}, off={phi_offset})")
                return test_action

            if v_scale == 1.0 and phi_offset == 0:
                safe_action = test_action

        if safe_action is not None:
            print("[Protector] 保守执行原计划")
            return safe_action

        # === 兜底防守 ===
        print("[Protector] 🛡️ 启动防守模式")

        nearest_target = None
        min_dist = 100
        cue_pos = balls['cue'].state.rvw[0]

        # 🔥 修复：防守时也要排除黑8（如果不能打）
        candidates = []
        for bid in my_targets:
            if bid == '8' and not can_shoot_8:
                continue
            if bid in balls and balls[bid].state.s != 4:
                candidates.append(bid)

        if not candidates:
            print("[Protector] ⚠️ 无合法目标球，尝试随机动作")
            return self._random_action()

        for tid in candidates:
            t_pos = balls[tid].state.rvw[0]
            d = self._distance(cue_pos, t_pos)
            if d < min_dist:
                min_dist = d
                nearest_target = tid

        if nearest_target:
            t_pos = balls[nearest_target].state.rvw[0]
            vec = t_pos - cue_pos
            phi = self._angle_to_phi(self._normalize(vec))
            print(f"[Protector] 防守目标：{nearest_target}，距离：{min_dist:.2f}m")
            return {'V0': 2.5, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}

        return action

    # ==================== 主决策函数 ====================
    def decision(self, balls, my_targets, table):
        try:
            # 0. 开球检测
            balls_on_table = [b for k, b in balls.items() if k != 'cue' and b.state.s != 4]
            if len(balls_on_table) == 15:
                print("[Protector] 🎱 开球")
                return self.get_break_shot(balls)

            # 🔥 关键修复：保存原始目标球列表
            original_targets = list(my_targets)  # 深拷贝，保留原始值

            # 检查是否需要切换到黑8
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if not remaining:
                my_targets = ['8']
                print("[Protector] ⭐ 己方球已清空，切换至黑8模式")

            # 显式打印当前状态
            can_shoot_8 = self._check_can_shoot_8(balls, original_targets)
            print(
                f"[Protector] 当前状态: original_targets={original_targets}, my_targets={my_targets}, can_shoot_8={can_shoot_8}")

            # 1. 选球（传入原始目标球）
            choice = self._choose_best_target(balls, my_targets, table, original_targets)
            if not choice:
                print("[Protector] 无可选目标，启动兜底")
                return self._validate_and_adjust({'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0},
                                                 balls, table, my_targets, original_targets)

            tid, pid = choice
            cue_pos = balls['cue'].state.rvw[0]
            target_pos = balls[tid].state.rvw[0]
            pocket_pos = table.pockets[pid].center

            print(f"[Protector] 选择目标：{tid} → 袋口：{pid}")

            # 2. 生成动作
            geo_action = self._geometric_shot(cue_pos, target_pos, pocket_pos)
            cut_angle = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos)
            obstruction = self._count_obstructions(balls, cue_pos, target_pos, exclude_ids=['cue', tid])

            final_action = geo_action
            if cut_angle > 10 or obstruction > 0:
                print(f"[Protector] 优化击球 (切角{cut_angle:.1f}°, 遮挡{obstruction})")
                final_action = self._optimized_search(geo_action, balls, my_targets, table, original_targets)

            # 3. 验证
            final_action = self._validate_and_adjust(final_action, balls, table, my_targets, original_targets)

            return final_action

        except Exception as e:
            print(f"[Protector] 异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()