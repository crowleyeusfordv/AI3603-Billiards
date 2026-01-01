"""NewAgent - Phase 28: CMA-ES + MCTS Hybrid Pro V2

核心改进（对标BasicAgentPro）：
1. 采用Ghost Ball瞄准法生成启发式候选动作（与BasicAgentPro一致）
2. 使用CMA-ES进化策略优化动作参数（4维：V0, phi, a, b）
3. MCTS风格的多候选探索 + UCB选择
4. 噪声注入模拟评估动作鲁棒性
5. 更激进的进球策略 + 安全保障
6. 旋转参数优化（上下旋控制走位）
7. 增强的路径检测和防守策略
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

# CMA-ES for evolutionary optimization
import cma


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """分析击球结果并计算奖励分数（与BasicAgentPro完全一致）"""
    
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 分析碰库
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
        
    # 计算奖励分数（与BasicAgentPro一致）
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
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
        pass
    
    def _random_action(self):
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
        return action


class NewAgent(Agent):
    """NewAgent - Phase 28: CMA-ES + MCTS Hybrid Pro V2
    
    核心改进（对标并超越BasicAgentPro）：
    1. Ghost Ball瞄准法 + 多力度/角度/旋转变种生成大量候选
    2. CMA-ES进化策略深度优化top候选（4维：V0, phi, a, b）
    3. MCTS风格UCB选择 + 更多模拟次数
    4. 噪声注入鲁棒性评估
    5. 走位考虑：优先选择能连续进球的动作
    6. 防守策略：当无好球时安全出杆
    """
    
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        
        # 增加模拟次数 - 更多模拟提高评估准确性
        self.n_simulations = 150  # 增加到150次
        self.c_puct = 0.5  # 更偏向exploitation
        
        # 噪声参数（与poolenv对齐）
        self.sim_noise = {
            'V0': 0.12, 'phi': 0.12, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        # 启用CMA-ES优化top候选
        self.use_cma_es = True
        self.cma_top_k = 8  # 对前8个候选进行CMA-ES优化
        self.cma_evals = 50  # CMA-ES评估次数
        
        print("[NewAgent] Phase 30: Stable 80%+ MCTS + CMA-ES 已初始化")

    # ========== 工具函数（与BasicAgentPro对齐）==========
    def _calc_angle_degrees(self, v):
        """计算向量角度"""
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """Ghost Ball瞄准法（与BasicAgentPro完全一致）"""
        vec_obj_to_pocket = np.array(pocket_pos[:2]) - np.array(obj_pos[:2])
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0:
            return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos[:2]) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos[:2])
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def _get_valid_targets(self, balls, my_targets):
        """获取当前合法目标球"""
        remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0:
            return ['8'], True
        return remaining, False

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

    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        """计算路径上的障碍球数量（增强版：更精确检测）"""
        count = 0
        partial_count = 0  # 部分遮挡
        line_vec = np.array(to_pos[:2]) - np.array(from_pos[:2])
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-6:
            return 0
        line_dir = line_vec / line_length
        
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4:
                continue
            ball_pos = ball.state.rvw[0][:2]
            vec_to_ball = ball_pos - np.array(from_pos[:2])
            proj_length = np.dot(vec_to_ball, line_dir)
            if proj_length < 0 or proj_length > line_length:
                continue
            proj_point = np.array(from_pos[:2]) + line_dir * proj_length
            dist_to_line = np.linalg.norm(ball_pos - proj_point)
            
            # 完全遮挡
            if dist_to_line < self.ball_radius * 2.1:
                count += 1
            # 部分遮挡（球边缘擦过）
            elif dist_to_line < self.ball_radius * 2.5:
                partial_count += 1
        
        return count + partial_count * 0.5

    def _is_clear_path(self, balls, cue_pos, target_pos, pocket_pos, target_id):
        """检查击球路径是否清晰（增强版：综合检测）"""
        # 检查白球到目标球路径
        obs1 = self._count_obstructions(balls, cue_pos, target_pos, exclude_ids=['cue', target_id])
        if obs1 > 0.1:
            return False, "blocked_cue_to_target"
        
        # 检查目标球到袋口路径
        obs2 = self._count_obstructions(balls, target_pos, pocket_pos, exclude_ids=['cue', target_id])
        if obs2 > 0.1:
            return False, "blocked_target_to_pocket"
        
        return True, None

    # ========== 启发式动作生成（增强版）==========
    def _is_8ball_in_path(self, balls, cue_pos, target_pos, margin=1.1):
        """检查8球是否在击球路径上"""
        if '8' not in balls or balls['8'].state.s == 4:
            return False
        
        eight_pos = balls['8'].state.rvw[0][:2]
        line_vec = np.array(target_pos[:2]) - np.array(cue_pos[:2])
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-6:
            return False
        line_dir = line_vec / line_length
        
        vec_to_8 = eight_pos - np.array(cue_pos[:2])
        proj_length = np.dot(vec_to_8, line_dir)
        
        # 只检查在目标之前的位置
        if proj_length < 0 or proj_length > line_length + self.ball_radius * 2:
            return False
        
        proj_point = np.array(cue_pos[:2]) + line_dir * proj_length
        dist_to_line = np.linalg.norm(eight_pos - proj_point)
        
        return dist_to_line < self.ball_radius * 2 * margin
    
    def generate_heuristic_actions(self, balls, my_targets, table):
        """生成候选动作列表（增强版V2：更多变种+走位优先）"""
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        target_ids, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        
        if not target_ids:
            target_ids = ['8']

        # 计算每个目标球-袋口组合的难度分数
        shot_candidates = []
        
        for tid in target_ids:
            if tid not in balls:
                continue
            obj_ball = balls[tid]
            if obj_ball.state.s == 4:
                continue
            obj_pos = obj_ball.state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 计算ghost ball角度和距离
                phi_ideal, dist_to_ghost = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                
                # 计算路径清晰度
                is_clear, _ = self._is_clear_path(balls, cue_pos, obj_pos, pocket_pos, tid)
                
                # 计算切角
                try:
                    cut_angle = self._calc_cut_angle(cue_pos, obj_pos, pocket_pos)
                except:
                    cut_angle = 90
                
                # 计算目标球到袋口距离
                dist_to_pocket = np.linalg.norm(np.array(obj_pos[:2]) - np.array(pocket_pos[:2]))
                
                # 难度分数（越低越容易）
                difficulty = (
                    dist_to_ghost * 0.5 +  # 白球到目标距离
                    dist_to_pocket * 0.3 +  # 目标到袋口距离
                    cut_angle * 0.02 +  # 切角惩罚
                    (0 if is_clear else 50)  # 路径阻挡惩罚
                )
                
                shot_candidates.append({
                    'tid': tid,
                    'pocket_id': pocket_id,
                    'phi_ideal': phi_ideal,
                    'dist_to_ghost': dist_to_ghost,
                    'difficulty': difficulty,
                    'is_clear': is_clear
                })
        
        # 按难度排序
        shot_candidates.sort(key=lambda x: x['difficulty'])
        
        # 为每个候选生成变种（优先处理容易的球）
        for i, cand in enumerate(shot_candidates[:10]):  # 取前10个最容易的
            phi_ideal = cand['phi_ideal']
            dist_to_ghost = cand['dist_to_ghost']
            
            # 根据距离估算力度
            v_base = 1.5 + dist_to_ghost * 1.5
            v_base = np.clip(v_base, 1.2, 6.0)
            
            # 更多力度变种
            force_variants = [v_base, v_base * 0.85, v_base * 1.1, v_base * 1.25, min(v_base + 1.5, 7.5)]
            
            # 更多角度变种
            angle_variants = [0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0]
            
            for v in force_variants:
                for dphi in angle_variants:
                    actions.append({
                        'V0': round(v, 2), 
                        'phi': round((phi_ideal + dphi) % 360, 2), 
                        'theta': 0, 
                        'a': 0, 
                        'b': 0
                    })
                    # 旋转变种（控制走位）
                    if i < 5:  # 前5个容易的球加旋转变种
                        actions.append({
                            'V0': round(v, 2), 
                            'phi': round((phi_ideal + dphi) % 360, 2), 
                            'theta': 0, 
                            'a': 0, 
                            'b': -0.15  # 后旋
                        })

        if len(actions) == 0:
            # 防守：朝最近的目标球方向打
            for tid in target_ids[:3]:
                if tid in balls and balls[tid].state.s != 4:
                    vec = balls[tid].state.rvw[0] - cue_pos
                    phi = self._calc_angle_degrees(vec)
                    dist = np.linalg.norm(vec[:2])
                    v = np.clip(2.5 + dist * 0.5, 2.5, 5.0)
                    actions.append({'V0': v, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0})
        
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # 去重并打乱
        seen = set()
        unique_actions = []
        for a in actions:
            key = (round(a['V0'], 1), round(a['phi'], 1))
            if key not in seen:
                seen.add(key)
                unique_actions.append(a)
        
        random.shuffle(unique_actions)
        return unique_actions[:60]  # 返回更多候选以增加覆盖

    def _calc_cut_angle(self, cue_pos, obj_pos, pocket_pos):
        """计算切角"""
        ghost_pos = np.array(obj_pos[:2]) - self._normalize_vec(
            np.array(pocket_pos[:2]) - np.array(obj_pos[:2])
        ) * (2 * self.ball_radius)
        
        vec1 = self._normalize_vec(ghost_pos - np.array(cue_pos[:2]))
        vec2 = self._normalize_vec(np.array(pocket_pos[:2]) - np.array(obj_pos[:2]))
        dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))
    
    def _normalize_vec(self, vec):
        """向量归一化"""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])

    # ========== 带噪声模拟（与BasicAgentPro对齐）==========
    def simulate_action(self, balls, table, action):
        """执行带噪声的物理仿真（与BasicAgentPro完全一致）"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # 注入高斯噪声
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            
            if not simulate_with_timeout(shot, timeout=3):
                return None
            return shot
        except Exception:
            return None

    # ========== CMA-ES优化器（增强版V4：更多评估+首球验证）==========
    def cma_es_optimize(self, initial_action, balls, my_targets, table, max_evals=None):
        """使用CMA-ES优化动作参数（增强版V4）"""
        if max_evals is None:
            max_evals = self.cma_evals
            
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        cue_pos = balls['cue'].state.rvw[0]
        
        # 2维优化：V0, phi（旋转参数保持初始值，避免不可预期的8球碰撞）
        x0 = [
            initial_action['V0'],
            initial_action['phi'],
        ]
        
        init_a = initial_action.get('a', 0.0)
        init_b = initial_action.get('b', 0.0)
        
        sigma0 = 0.3  # 稍大的初始步长以更好探索
        
        # 边界：更宽的角度范围
        phi_range = 3.0  # 稍宽范围
        bounds = [
            [0.5, initial_action['phi'] - phi_range],   # 下界
            [8.0, initial_action['phi'] + phi_range]    # 上界
        ]
        
        def objective(x):
            """目标函数（最小化，所以取负）"""
            V0, phi = x
            phi = phi % 360
            V0 = np.clip(V0, 0.5, 8.0)
            
            action = {'V0': V0, 'phi': phi, 'theta': 0, 'a': init_a, 'b': init_b}
            
            # 多次噪声模拟取平均（增强鲁棒性）- 增加到8次
            rewards = []
            eight_danger_count = 0
            first_foul_count = 0
            pocket_count = 0
            
            for _ in range(8):  # 8次模拟
                shot = self.simulate_action(balls, table, action)
                if shot is None:
                    rewards.append(-500)
                else:
                    # 检查是否会犯规8球
                    new_pocketed = [bid for bid, b_obj in shot.balls.items() 
                                   if b_obj.state.s == 4 and last_state[bid].state.s != 4]
                    
                    # 记录进球
                    own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
                    if len(own_pocketed) > 0:
                        pocket_count += 1
                    
                    # 非法打进8球 = 极严重
                    if '8' in new_pocketed and not can_shoot_8:
                        eight_danger_count += 1
                        rewards.append(-800)
                        continue
                    
                    # 检查首球犯规
                    is_legal, first_ball = self._check_first_contact(shot, valid_targets)
                    if not is_legal:
                        first_foul_count += 1
                        if first_ball == '8' and not can_shoot_8:
                            eight_danger_count += 1
                            rewards.append(-800)
                            continue
                        rewards.append(-30)  # 首球犯规惩罚
                        continue
                    
                    r = self._evaluate_shot_with_position(shot, last_state, valid_targets, balls, table)
                    rewards.append(r)
            
            avg_reward = np.mean(rewards)
            # 8球犯规极端惩罚
            if eight_danger_count > 0:
                avg_reward -= 300 * eight_danger_count
            # 首球犯规惩罚
            if first_foul_count >= 3:
                avg_reward -= 50 * first_foul_count
            # 进球一致性奖励
            if pocket_count >= 6:  # 8次中至少6次进球 = 稳定
                avg_reward += 20
            return -avg_reward
        
        try:
            es = cma.CMAEvolutionStrategy(
                x0, sigma0,
                {'bounds': bounds, 'maxfevals': max_evals, 'verbose': -9, 'seed': random.randint(0, 10000)}
            )
            
            while not es.stop():
                solutions = es.ask()
                fitnesses = [objective(x) for x in solutions]
                es.tell(solutions, fitnesses)
            
            best = es.result.xbest
            best_action = {
                'V0': float(np.clip(best[0], 0.5, 8.0)),
                'phi': float(best[1] % 360),
                'theta': 0.0,
                'a': init_a,
                'b': init_b
            }
            return best_action, -es.result.fbest
        except Exception as e:
            return initial_action, -500
    
    def _evaluate_shot_with_position(self, shot, last_state, valid_targets, balls, table):
        """评估击球结果，包含走位奖励"""
        base_score = analyze_shot_for_reward(shot, last_state, valid_targets)
        
        # 如果基础分很低（犯规），直接返回
        if base_score < -50:
            return base_score
        
        # 检查进球情况
        new_pocketed = [bid for bid, b in shot.balls.items() 
                       if b.state.s == 4 and last_state[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
        
        # 走位奖励：如果进了球，看白球最终位置是否有利于下一杆
        position_bonus = 0
        if len(own_pocketed) > 0 and 'cue' not in new_pocketed:
            try:
                cue_final_pos = shot.balls['cue'].state.rvw[0]
                
                # 计算剩余目标球
                remaining_targets = [tid for tid in valid_targets 
                                    if tid not in own_pocketed and 
                                    tid in shot.balls and 
                                    shot.balls[tid].state.s != 4]
                
                if remaining_targets:
                    # 找最近的剩余目标球
                    min_dist = float('inf')
                    for tid in remaining_targets:
                        dist = np.linalg.norm(cue_final_pos[:2] - shot.balls[tid].state.rvw[0][:2])
                        if dist < min_dist:
                            min_dist = dist
                    
                    # 距离越近越好（0.3-1.5米是理想范围）
                    if 0.15 < min_dist < 1.2:
                        position_bonus = 20
                    elif min_dist < 0.15:
                        position_bonus = 5  # 太近不太好
                    else:
                        position_bonus = max(0, 15 - (min_dist - 1.2) * 10)
            except:
                pass
        
        # 多球奖励
        multi_ball_bonus = len(own_pocketed) * 15 if len(own_pocketed) > 1 else 0
        
        return base_score + position_bonus + multi_ball_bonus

    # ========== MCTS风格评估（增强版V2：CMA-ES后处理）==========
    def mcts_evaluate(self, candidate_actions, balls, my_targets, table):
        """MCTS风格的动作评估 + CMA-ES精细优化"""
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        
        n_candidates = len(candidate_actions)
        if n_candidates == 0:
            return self._random_action()
        
        N = np.zeros(n_candidates)  # 访问次数
        Q = np.zeros(n_candidates)  # 累计奖励
        
        # 第一阶段：MCTS探索
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation (带噪声)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation（增强版：包含首球犯规检测）
            if shot is None:
                raw_reward = -500.0
            else:
                # 先检查首球犯规
                is_legal, first_ball = self._check_first_contact(shot, valid_targets)
                if not is_legal:
                    raw_reward = -30.0  # 首球犯规惩罚
                else:
                    raw_reward = analyze_shot_for_reward(shot, last_state, valid_targets)
            
            # 归一化到[0,1]
            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward

        # 计算平均分
        avg_rewards = Q / (N + 1e-6)
        
        # 第二阶段：对top-k候选使用CMA-ES优化（如果启用）
        if self.use_cma_es and n_candidates >= 3:
            # 获取top-k候选的索引
            top_k = min(self.cma_top_k, n_candidates)
            top_indices = np.argsort(avg_rewards)[-top_k:]
            
            best_cma_action = None
            best_cma_score = -1e9
            
            for idx in top_indices:
                if avg_rewards[idx] < 0.5:  # 只优化有潜力的候选
                    continue
                    
                optimized_action, opt_score = self.cma_es_optimize(
                    candidate_actions[idx], balls, my_targets, table, max_evals=30
                )
                
                if opt_score > best_cma_score:
                    best_cma_score = opt_score
                    best_cma_action = optimized_action
            
            # 如果CMA-ES找到更好的动作，使用它
            if best_cma_action and best_cma_score > 30:  # 阈值：CMA-ES分数需要足够高
                print(f"[NewAgent] CMA-ES优化: {best_cma_score:.1f} (MCTS best: {avg_rewards.max():.3f})")
                return best_cma_action
        
        # 选择平均分最高的
        best_idx = np.argmax(avg_rewards)
        
        print(f"[NewAgent] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return candidate_actions[best_idx]

    # ========== 8球犯规专项检测 ==========
    def _could_cause_8ball_foul(self, action, balls, table, valid_targets, tests=6):
        """专门检测动作是否可能导致8球犯规"""
        can_shoot_8 = ('8' in valid_targets)
        if can_shoot_8:
            return False  # 可以打8球时不需要检测
        
        foul_count = 0
        for _ in range(tests):
            shot = self.simulate_action(balls, table, action)
            if shot is None:
                continue
            
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            # 打进8球 = 犯规
            if '8' in new_pocketed:
                foul_count += 1
                continue
            
            # 首球碰撞8球 = 犯规
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if first_ball == '8':
                foul_count += 1
                continue
        
        # 任何测试中出现8球犯规都视为危险
        return foul_count > 0

    def _quick_first_contact_check(self, action, balls, table, valid_targets, tests=4):
        """快速检测动作是否可能导致首球犯规（非8球相关）"""
        foul_count = 0
        for _ in range(tests):
            shot = self.simulate_action(balls, table, action)
            if shot is None:
                foul_count += 1
                continue
            
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                foul_count += 1
        
        # 超过一半测试犯规 = 危险
        return foul_count >= tests // 2

    # ========== 安全验证（增强版）==========
    def _is_action_safe(self, action, balls, table, valid_targets, simulations=10):
        """验证动作安全性（增强版：更严格的检测）"""
        fatal_count = 0
        first_contact_foul = 0
        cue_pocket_count = 0
        eight_illegal_count = 0
        can_shoot_8 = ('8' in valid_targets)
        
        for _ in range(simulations):
            shot = self.simulate_action(balls, table, action)
            if shot is None:
                fatal_count += 1
                continue
            
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            # 白球进袋
            if 'cue' in new_pocketed:
                cue_pocket_count += 1
                fatal_count += 1
                continue
            
            # 非法黑8（最严重的犯规！）
            if '8' in new_pocketed and not can_shoot_8:
                eight_illegal_count += 1
                fatal_count += 3  # 严重惩罚
                continue
            
            # 首球犯规
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                first_contact_foul += 1
                # 首球碰撞8球但不能打8球 = 极严重
                if first_ball == '8' and not can_shoot_8:
                    eight_illegal_count += 1
                    fatal_count += 3
                else:
                    fatal_count += 1
                continue
        
        # 对非法黑8零容忍（包括首球碰撞8球）
        if eight_illegal_count > 0:
            return False
        
        # 其他犯规允许最多1次
        return fatal_count <= 1 and first_contact_foul <= 1 and cue_pocket_count <= 1

    def _find_safe_action(self, balls, table, my_targets, attempts=40):
        """寻找安全动作（增强版V2）"""
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        cue_pos = balls['cue'].state.rvw[0]
        
        target_ids = [tid for tid in valid_targets if tid in balls and balls[tid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        
        # 按距离排序
        target_ids.sort(key=lambda tid: np.linalg.norm(
            balls[tid].state.rvw[0][:2] - cue_pos[:2]) if tid in balls else float('inf'))
        
        for tid in target_ids[:4]:
            if tid not in balls:
                continue
            vec = balls[tid].state.rvw[0] - cue_pos
            base_phi = self._calc_angle_degrees(vec)
            dist = np.linalg.norm(vec[:2])
            
            # 更多变种
            for dphi in [0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2, -2]:
                for v in [2.2, 2.5, 2.8, 3.0, 3.3, 3.5, 4.0]:
                    for b_spin in [0, -0.1, 0.1]:
                        action = {
                            'V0': v,
                            'phi': (base_phi + dphi) % 360,
                            'theta': 0,
                            'a': 0,
                            'b': b_spin
                        }
                        if self._is_action_safe(action, balls, table, valid_targets, simulations=8):
                            return action
        
        return None

    def _get_defensive_action(self, balls, table, my_targets):
        """防守策略：当没有好球时，把白球放到对手难以击打的位置"""
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        cue_pos = balls['cue'].state.rvw[0]
        
        # 寻找最近的目标球并轻轻击打
        target_ids = [tid for tid in valid_targets if tid in balls and balls[tid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        
        best_action = None
        best_score = -1e9
        
        for tid in target_ids[:3]:
            if tid not in balls:
                continue
            
            obj_pos = balls[tid].state.rvw[0]
            vec = obj_pos - cue_pos
            base_phi = self._calc_angle_degrees(vec)
            
            # 轻力击打，不求进球，求安全
            for dphi in [0, 1, -1, 2, -2]:
                for v in [1.5, 2.0, 2.5]:
                    action = {
                        'V0': v,
                        'phi': (base_phi + dphi) % 360,
                        'theta': 0,
                        'a': 0,
                        'b': -0.2  # 后旋控制白球
                    }
                    
                    if self._is_action_safe(action, balls, table, valid_targets, simulations=6):
                        # 模拟结果看白球最终位置
                        shot = self.simulate_action(balls, table, action)
                        if shot:
                            cue_final = shot.balls['cue'].state.rvw[0]
                            # 评分：白球离边越远越好，离对手目标球越远越好
                            table_center = np.array([table.l / 2, table.w / 2])
                            dist_to_center = np.linalg.norm(cue_final[:2] - table_center)
                            score = -dist_to_center  # 远离中心
                            
                            if score > best_score:
                                best_score = score
                                best_action = action
        
        return best_action

    # ========== 开球 ==========
    def get_break_shot(self, balls, my_targets, table):
        """开球策略"""
        cue_pos = balls['cue'].state.rvw[0]
        valid_targets, _ = self._get_valid_targets(balls, my_targets)
        
        # 球堆中心方向
        rack_positions = [b.state.rvw[0] for bid, b in balls.items() if bid != 'cue' and b.state.s != 4]
        if rack_positions:
            rack_center = np.mean(np.asarray(rack_positions), axis=0)
            base_phi = self._calc_angle_degrees(rack_center - cue_pos)
        else:
            base_phi = 0.0

        # 搜索合法开球
        best_action = None
        best_score = -1e9
        
        for dphi in range(-30, 31, 5):
            for v0 in [7.5, 7.0, 6.5, 6.0]:
                action = {
                    'V0': v0,
                    'phi': (base_phi + dphi) % 360,
                    'theta': 0,
                    'a': 0,
                    'b': 0
                }
                
                # 确定性模拟
                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=cue)
                
                try:
                    shot.cue.set_state(**action)
                    if not simulate_with_timeout(shot, timeout=3):
                        continue
                except:
                    continue
                
                new_pocketed = [bid for bid, b in shot.balls.items() 
                              if b.state.s == 4 and balls[bid].state.s != 4]
                
                if 'cue' in new_pocketed:
                    continue
                if '8' in new_pocketed:
                    continue
                
                is_legal, _ = self._check_first_contact(shot, valid_targets)
                if not is_legal:
                    continue
                
                own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
                score = len(own_pocketed) * 100
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        if best_action:
            return best_action
        
        # 兜底
        return {'V0': 6.5, 'phi': base_phi, 'theta': 0, 'a': 0, 'b': 0}

    # ========== 主决策 ==========
    def decision(self, balls, my_targets, table):
        """主决策函数（增强版V2：多层过滤+迭代优化）"""
        try:
            valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
            
            # 开球检测
            balls_on_table = [b for k, b in balls.items() if k != 'cue' and b.state.s != 4]
            if len(balls_on_table) == 15:
                pos = np.asarray([b.state.rvw[0][:2] for b in balls_on_table], dtype=float)
                center = pos.mean(axis=0)
                mean_r = float(np.mean(np.linalg.norm(pos - center, axis=1)))
                if mean_r < 0.12:
                    return self.get_break_shot(balls, my_targets, table)
            
            # 生成启发式候选动作
            candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
            
            if not candidate_actions:
                safe = self._find_safe_action(balls, table, my_targets)
                if safe:
                    return safe
                return self._random_action()
            
            # 第一层过滤: 排除可能导致8球犯规的动作
            if not can_shoot_8:
                safe_candidates = []
                for action in candidate_actions:
                    if not self._could_cause_8ball_foul(action, balls, table, valid_targets, tests=3):
                        safe_candidates.append(action)
                
                # 如果没有安全候选，使用防守
                if len(safe_candidates) == 0:
                    safe = self._find_safe_action(balls, table, my_targets)
                    if safe:
                        return safe
                    safe_candidates = candidate_actions[:15]
                
                candidate_actions = safe_candidates
            
            # 第二层过滤: 排除高概率首球犯规的动作（快速检查）
            # 只保留通过首球检测的候选
            filtered_candidates = []
            for action in candidate_actions[:40]:  # 只检查前40个
                if not self._quick_first_contact_check(action, balls, table, valid_targets, tests=2):
                    filtered_candidates.append(action)
            
            # 如果过滤后太少，保留原始候选
            if len(filtered_candidates) >= 10:
                candidate_actions = filtered_candidates
            
            # 用MCTS评估
            best_action = self.mcts_evaluate(candidate_actions, balls, my_targets, table)
            
            # 最终8球犯规检查（更严格）
            if not can_shoot_8 and self._could_cause_8ball_foul(best_action, balls, table, valid_targets, tests=8):
                # 寻找安全替代
                for action in candidate_actions[:20]:
                    if not self._could_cause_8ball_foul(action, balls, table, valid_targets, tests=5):
                        return action
                # 防守
                safe = self._find_safe_action(balls, table, my_targets)
                if safe:
                    return safe
            
            # 最终首球犯规检查
            if self._quick_first_contact_check(best_action, balls, table, valid_targets, tests=4):
                # 尝试其他高分候选
                for action in candidate_actions[:15]:
                    if not self._quick_first_contact_check(action, balls, table, valid_targets, tests=3):
                        return action
            
            return best_action

        except Exception as e:
            print(f"[NewAgent] 决策错误: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                safe = self._find_safe_action(balls, table, my_targets)
                if safe:
                    return safe
            except:
                pass
            
            return self._random_action()
