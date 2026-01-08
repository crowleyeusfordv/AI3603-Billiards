"""NewAgent - Phase 28: CMA-ES + MCTS Hybrid Pro V2 (多进程优化版)

核心改进（对标BasicAgentPro）：
1. 采用Ghost Ball瞄准法生成启发式候选动作（与BasicAgentPro一致）
2. 使用CMA-ES进化策略优化动作参数（4维：V0, phi, a, b）
3. MCTS风格的多候选探索 + UCB选择
4. 噪声注入模拟评估动作鲁棒性
5. 更激进的进球策略 + 安全保障
6. 旋转参数优化（上下旋控制走位）
7. 增强的路径检测和防守策略
8. **多进程并行物理模拟加速**
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
import multiprocessing as mp
from functools import partial

# CMA-ES for evolutionary optimization
import cma


# ============ 多进程配置 ============
# 全局进程池（懒加载）
_global_pool = None
_pool_size = max(1, mp.cpu_count() - 1)  # 留一个核心给主进程

def get_pool():
    """获取全局进程池（懒加载）"""
    global _global_pool
    if _global_pool is None:
        _global_pool = mp.Pool(processes=_pool_size, initializer=_init_worker)
    return _global_pool

def _init_worker():
    """初始化worker进程"""
    # 每个worker使用不同的随机种子
    seed = os.getpid() + int(random.random() * 10000)
    random.seed(seed)
    np.random.seed(seed)


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


# ============ 序列化辅助函数 ============
def serialize_balls(balls):
    """序列化球状态为可pickle的dict"""
    data = {}
    for bid, ball in balls.items():
        data[bid] = {
            'rvw': ball.state.rvw.copy(),
            's': ball.state.s,
            't': ball.state.t,
            'params': {
                'm': ball.params.m,
                'R': ball.params.R,
            }
        }
    return data

def deserialize_balls(data):
    """从序列化数据重建球对象"""
    balls = {}
    for bid, info in data.items():
        ball = pt.Ball.create(bid, xy=info['rvw'][0][:2])
        ball.state.rvw = info['rvw']
        ball.state.s = info['s']
        ball.state.t = info['t']
        balls[bid] = ball
    return balls

def serialize_table(table):
    """序列化球桌（简化版：只传必要参数）"""
    return {
        'l': table.l,
        'w': table.w,
        'table_type': 'pocket'  # 假设都是标准袋球台
    }

def deserialize_table(data):
    """重建球桌"""
    return pt.Table.from_table_specs(PocketTableSpecs())


# ============ 并行模拟Worker函数 ============
def _simulate_single_action_worker(args):
    """Worker函数：执行单次带噪声的物理模拟
    
    Args:
        args: (balls_data, table_data, action, sim_noise, worker_seed)
    
    Returns:
        dict: 模拟结果
    """
    balls_data, table_data, action, sim_noise, worker_seed = args
    
    # 设置随机种子
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    try:
        # 重建环境
        balls = deserialize_balls(balls_data)
        table = deserialize_table(table_data)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=table, balls=balls, cue=cue)
        
        # 注入高斯噪声
        noisy_V0 = np.clip(action['V0'] + np.random.normal(0, sim_noise['V0']), 0.5, 8.0)
        noisy_phi = (action['phi'] + np.random.normal(0, sim_noise['phi'])) % 360
        noisy_theta = np.clip(action['theta'] + np.random.normal(0, sim_noise['theta']), 0, 90)
        noisy_a = np.clip(action['a'] + np.random.normal(0, sim_noise['a']), -0.5, 0.5)
        noisy_b = np.clip(action['b'] + np.random.normal(0, sim_noise['b']), -0.5, 0.5)
        
        cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
        
        # 执行模拟（带超时）
        if not simulate_with_timeout(shot, timeout=3):
            return {'success': False, 'error': 'timeout'}
        
        # 提取结果
        result = {
            'success': True,
            'balls_final': {},
            'first_contact': None,
            'events_summary': []
        }
        
        # 记录球的最终状态
        for bid, ball in shot.balls.items():
            result['balls_final'][bid] = {
                'rvw': ball.state.rvw.copy(),
                's': ball.state.s
            }
        
        # 分析首球碰撞
        valid_ball_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            
            # 记录事件摘要
            result['events_summary'].append({'type': et, 'ids': ids})
            
            # 找首球碰撞
            if result['first_contact'] is None:
                if 'cushion' not in et and 'pocket' not in et and 'cue' in ids:
                    other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                    if other_ids:
                        result['first_contact'] = other_ids[0]
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


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
    """NewAgent - Phase 28: CMA-ES + MCTS Hybrid Pro V2 (多进程优化版)
    
    核心改进（对标并超越BasicAgentPro）：
    1. Ghost Ball瞄准法 + 多力度/角度/旋转变种生成大量候选
    2. CMA-ES进化策略深度优化top候选（4维：V0, phi, a, b）
    3. MCTS风格UCB选择 + 更多模拟次数
    4. 噪声注入鲁棒性评估
    5. 走位考虑：优先选择能连续进球的动作
    6. 防守策略：当无好球时安全出杆
    7. **多进程并行物理模拟，大幅减少决策时间**
    """
    
    def __init__(self, use_multiprocess=True, n_workers=None):
        super().__init__()
        self.ball_radius = 0.028575
        
        # 增加模拟次数 - 更多模拟提高评估准确性
        self.n_simulations = 250  # 增加到250次
        self.c_puct = 0.5  # 更偏向exploitation
        
        # 噪声参数（与poolenv对齐）
        self.sim_noise = {
            'V0': 0.12, 'phi': 0.12, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        # 启用CMA-ES优化top候选
        self.use_cma_es = True
        self.cma_top_k = 8  # 对前8个候选进行CMA-ES优化
        self.cma_evals = 80  # CMA-ES评估次数
        
        # 多进程配置
        # 检测是否在子进程中运行（daemon进程不能创建子进程）
        try:
            current_process = mp.current_process()
            is_daemon = current_process.daemon
        except:
            is_daemon = False
        
        # 如果是daemon进程，禁用多进程
        if is_daemon:
            self.use_multiprocess = False
            self.n_workers = 1
        else:
            self.use_multiprocess = use_multiprocess
            self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        self._local_pool = None
        
        print(f"[NewAgent] Phase 30: Stable 80%+ MCTS + CMA-ES 已初始化 (多进程: {self.use_multiprocess}, Workers: {self.n_workers})")

    def _get_pool(self):
        """获取进程池"""
        if not self.use_multiprocess:
            return None
        if self._local_pool is None:
            self._local_pool = mp.Pool(processes=self.n_workers, initializer=_init_worker)
        return self._local_pool

    def __del__(self):
        """清理进程池"""
        if self._local_pool is not None:
            try:
                self._local_pool.close()
                self._local_pool.join()
            except:
                pass

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

    # ========== 带噪声模拟（支持多进程）==========
    def simulate_action(self, balls, table, action):
        """执行带噪声的物理仿真（单次，用于兼容原有代码）"""
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

    def batch_simulate_actions(self, balls, table, actions, n_simulations_per_action=1):
        """批量并行模拟多个动作
        
        Args:
            balls: 球状态
            table: 球桌
            actions: 动作列表
            n_simulations_per_action: 每个动作模拟次数
        
        Returns:
            list of list: 每个动作的模拟结果列表
        """
        if not self.use_multiprocess or len(actions) == 0:
            # 单进程模式
            results = []
            for action in actions:
                action_results = []
                for _ in range(n_simulations_per_action):
                    shot = self.simulate_action(balls, table, action)
                    if shot is None:
                        action_results.append({'success': False, 'error': 'simulation_failed'})
                    else:
                        # 转换为结果格式
                        result = self._extract_shot_result(shot, balls)
                        action_results.append(result)
                results.append(action_results)
            return results
        
        # 多进程模式
        try:
            balls_data = serialize_balls(balls)
            table_data = serialize_table(table)
            
            # 准备所有任务
            tasks = []
            base_seed = random.randint(0, 100000)
            task_idx = 0
            
            for action_idx, action in enumerate(actions):
                for sim_idx in range(n_simulations_per_action):
                    worker_seed = base_seed + task_idx
                    tasks.append((balls_data, table_data, action, self.sim_noise, worker_seed))
                    task_idx += 1
            
            # 并行执行
            pool = self._get_pool()
            all_results = pool.map(_simulate_single_action_worker, tasks)
            
            # 按动作分组结果
            results = []
            idx = 0
            for action_idx, action in enumerate(actions):
                action_results = []
                for sim_idx in range(n_simulations_per_action):
                    action_results.append(all_results[idx])
                    idx += 1
                results.append(action_results)
            
            return results
            
        except Exception as e:
            # 回退到单进程
            print(f"[NewAgent] 多进程模拟失败，回退单进程: {e}")
            return self.batch_simulate_actions.__wrapped__(self, balls, table, actions, n_simulations_per_action)
    
    def _extract_shot_result(self, shot, original_balls):
        """从shot对象提取结果（用于单进程模式）"""
        result = {
            'success': True,
            'balls_final': {},
            'first_contact': None,
            'events_summary': []
        }
        
        for bid, ball in shot.balls.items():
            result['balls_final'][bid] = {
                'rvw': ball.state.rvw.copy(),
                's': ball.state.s
            }
        
        valid_ball_ids = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'}
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            result['events_summary'].append({'type': et, 'ids': ids})
            
            if result['first_contact'] is None:
                if 'cushion' not in et and 'pocket' not in et and 'cue' in ids:
                    other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                    if other_ids:
                        result['first_contact'] = other_ids[0]
        
        return result

    def _analyze_result_for_reward(self, result, last_state_s, player_targets):
        """从模拟结果计算奖励（适用于并行模拟结果）"""
        if not result['success']:
            return -500
        
        balls_final = result['balls_final']
        first_contact = result['first_contact']
        
        # 找出新进袋的球
        new_pocketed = [bid for bid, info in balls_final.items() 
                       if info['s'] == 4 and last_state_s.get(bid, 0) != 4]
        
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        
        # 首球犯规检测
        foul_first_hit = False
        if first_contact is None:
            if len(last_state_s) > 2 or player_targets != ['8']:
                foul_first_hit = True
        elif first_contact not in player_targets:
            foul_first_hit = True
        
        # 碰库检测
        cue_hit_cushion = False
        target_hit_cushion = False
        for ev in result['events_summary']:
            if 'cushion' in ev['type']:
                if 'cue' in ev['ids']:
                    cue_hit_cushion = True
                if first_contact and first_contact in ev['ids']:
                    target_hit_cushion = True
        
        foul_no_rail = (len(new_pocketed) == 0 and first_contact is not None 
                       and not cue_hit_cushion and not target_hit_cushion)
        
        # 计算分数
        score = 0
        
        if cue_pocketed and eight_pocketed:
            score -= 500
        elif cue_pocketed:
            score -= 100
        elif eight_pocketed:
            is_targeting_eight = (len(player_targets) == 1 and player_targets[0] == "8")
            score += 150 if is_targeting_eight else -500
        
        if foul_first_hit:
            score -= 30
        if foul_no_rail:
            score -= 30
        
        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20
        
        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            score = 10
        
        return score

    # ========== CMA-ES优化器（多进程并行版）==========
    def cma_es_optimize(self, initial_action, balls, my_targets, table, max_evals=None):
        """使用CMA-ES优化动作参数（多进程并行版）"""
        if max_evals is None:
            max_evals = self.cma_evals
            
        last_state_s = {bid: ball.state.s for bid, ball in balls.items()}
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        
        # 2维优化：V0, phi
        x0 = [initial_action['V0'], initial_action['phi']]
        init_a = initial_action.get('a', 0.0)
        init_b = initial_action.get('b', 0.0)
        
        sigma0 = 0.3
        phi_range = 3.0
        bounds = [
            [0.5, initial_action['phi'] - phi_range],
            [8.0, initial_action['phi'] + phi_range]
        ]
        
        # 序列化数据（用于多进程）
        if self.use_multiprocess:
            balls_data = serialize_balls(balls)
            table_data = serialize_table(table)
        
        def objective_batch(solutions):
            """批量评估目标函数（支持并行）"""
            n_sims_per_solution = 8
            
            # 构建动作列表
            actions = []
            for x in solutions:
                V0, phi = x
                phi = phi % 360
                V0 = np.clip(V0, 0.5, 8.0)
                actions.append({'V0': V0, 'phi': phi, 'theta': 0, 'a': init_a, 'b': init_b})
            
            if self.use_multiprocess:
                # 并行模拟
                try:
                    tasks = []
                    base_seed = random.randint(0, 100000)
                    task_idx = 0
                    
                    for action in actions:
                        for _ in range(n_sims_per_solution):
                            worker_seed = base_seed + task_idx
                            tasks.append((balls_data, table_data, action, self.sim_noise, worker_seed))
                            task_idx += 1
                    
                    pool = self._get_pool()
                    raw_results = pool.map(_simulate_single_action_worker, tasks)
                    
                    # 分析结果
                    fitnesses = []
                    idx = 0
                    for action in actions:
                        rewards = []
                        eight_danger = 0
                        first_foul = 0
                        pocket_count = 0
                        
                        for _ in range(n_sims_per_solution):
                            result = raw_results[idx]
                            idx += 1
                            
                            if not result['success']:
                                rewards.append(-500)
                                continue
                            
                            # 检查8球犯规
                            new_pocketed = [bid for bid, info in result['balls_final'].items()
                                           if info['s'] == 4 and last_state_s.get(bid, 0) != 4]
                            
                            own_pocketed = [bid for bid in new_pocketed if bid in valid_targets]
                            if len(own_pocketed) > 0:
                                pocket_count += 1
                            
                            if '8' in new_pocketed and not can_shoot_8:
                                eight_danger += 1
                                rewards.append(-800)
                                continue
                            
                            # 检查首球犯规
                            first_contact = result['first_contact']
                            is_legal = (first_contact in valid_targets) if first_contact else False
                            
                            if not is_legal:
                                first_foul += 1
                                if first_contact == '8' and not can_shoot_8:
                                    eight_danger += 1
                                    rewards.append(-800)
                                    continue
                                rewards.append(-30)
                                continue
                            
                            r = self._analyze_result_for_reward(result, last_state_s, valid_targets)
                            rewards.append(r)
                        
                        avg_reward = np.mean(rewards)
                        if eight_danger > 0:
                            avg_reward -= 300 * eight_danger
                        if first_foul >= 3:
                            avg_reward -= 50 * first_foul
                        if pocket_count >= 6:
                            avg_reward += 20
                        
                        fitnesses.append(-avg_reward)  # 最小化
                    
                    return fitnesses
                    
                except Exception as e:
                    pass  # 回退到单进程
            
            # 单进程模式
            fitnesses = []
            for action in actions:
                rewards = []
                for _ in range(n_sims_per_solution):
                    shot = self.simulate_action(balls, table, action)
                    if shot is None:
                        rewards.append(-500)
                    else:
                        new_pocketed = [bid for bid, b in shot.balls.items()
                                       if b.state.s == 4 and last_state_s.get(bid, 0) != 4]
                        
                        if '8' in new_pocketed and not can_shoot_8:
                            rewards.append(-800)
                            continue
                        
                        is_legal, first_ball = self._check_first_contact(shot, valid_targets)
                        if not is_legal:
                            if first_ball == '8' and not can_shoot_8:
                                rewards.append(-800)
                                continue
                            rewards.append(-30)
                            continue
                        
                        r = analyze_shot_for_reward(shot, 
                            {bid: copy.deepcopy(b) for bid, b in balls.items()}, valid_targets)
                        rewards.append(r)
                
                fitnesses.append(-np.mean(rewards))
            
            return fitnesses
        
        try:
            es = cma.CMAEvolutionStrategy(
                x0, sigma0,
                {'bounds': bounds, 'maxfevals': max_evals, 'verbose': -9, 'seed': random.randint(0, 10000)}
            )
            
            while not es.stop():
                solutions = es.ask()
                fitnesses = objective_batch(solutions)
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

    # ========== MCTS风格评估（多进程并行版）==========
    def mcts_evaluate(self, candidate_actions, balls, my_targets, table):
        """MCTS风格的动作评估 + CMA-ES精细优化（多进程并行版）"""
        valid_targets, can_shoot_8 = self._get_valid_targets(balls, my_targets)
        
        n_candidates = len(candidate_actions)
        if n_candidates == 0:
            return self._random_action()
        
        # 记录原始状态（用于奖励计算）
        last_state_s = {bid: ball.state.s for bid, ball in balls.items()}
        
        # 计算每个候选的模拟次数分配
        sims_per_candidate = max(1, self.n_simulations // n_candidates)
        extra_sims = self.n_simulations % n_candidates
        
        # 批量并行模拟所有候选
        sim_counts = [sims_per_candidate + (1 if i < extra_sims else 0) for i in range(n_candidates)]
        
        # 使用多进程批量模拟
        if self.use_multiprocess:
            # 准备批量任务
            all_results = self._parallel_mcts_simulate(candidate_actions, sim_counts, balls, table, valid_targets, last_state_s)
        else:
            # 单进程模式
            all_results = []
            for idx, action in enumerate(candidate_actions):
                action_results = []
                for _ in range(sim_counts[idx]):
                    shot = self.simulate_action(balls, table, action)
                    if shot is None:
                        action_results.append(-500.0)
                    else:
                        is_legal, first_ball = self._check_first_contact(shot, valid_targets)
                        if not is_legal:
                            action_results.append(-30.0)
                        else:
                            action_results.append(analyze_shot_for_reward(shot, 
                                {bid: copy.deepcopy(b) for bid, b in balls.items()}, valid_targets))
                all_results.append(action_results)
        
        # 计算每个候选的平均分
        avg_rewards = []
        for action_results in all_results:
            if len(action_results) > 0:
                # 归一化到[0,1]
                normalized = [(r - (-500)) / 650.0 for r in action_results]
                normalized = [np.clip(r, 0.0, 1.0) for r in normalized]
                avg_rewards.append(np.mean(normalized))
            else:
                avg_rewards.append(0.0)
        
        avg_rewards = np.array(avg_rewards)
        
        # 第二阶段：对top-k候选使用CMA-ES优化（如果启用）
        if self.use_cma_es and n_candidates >= 3:
            top_k = min(self.cma_top_k, n_candidates)
            top_indices = np.argsort(avg_rewards)[-top_k:]
            
            best_cma_action = None
            best_cma_score = -1e9
            
            for idx in top_indices:
                if avg_rewards[idx] < 0.5:
                    continue
                    
                optimized_action, opt_score = self.cma_es_optimize(
                    candidate_actions[idx], balls, my_targets, table, max_evals=30
                )
                
                if opt_score > best_cma_score:
                    best_cma_score = opt_score
                    best_cma_action = optimized_action
            
            if best_cma_action and best_cma_score > 30:
                print(f"[NewAgent] CMA-ES优化: {best_cma_score:.1f} (MCTS best: {avg_rewards.max():.3f})")
                return best_cma_action
        
        best_idx = np.argmax(avg_rewards)
        print(f"[NewAgent] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return candidate_actions[best_idx]

    def _parallel_mcts_simulate(self, candidate_actions, sim_counts, balls, table, valid_targets, last_state_s):
        """并行执行MCTS模拟"""
        try:
            balls_data = serialize_balls(balls)
            table_data = serialize_table(table)
            
            # 准备所有任务
            tasks = []
            base_seed = random.randint(0, 100000)
            task_idx = 0
            task_mapping = []  # 记录任务与候选的对应关系
            
            for action_idx, action in enumerate(candidate_actions):
                for sim_idx in range(sim_counts[action_idx]):
                    worker_seed = base_seed + task_idx
                    tasks.append((balls_data, table_data, action, self.sim_noise, worker_seed))
                    task_mapping.append(action_idx)
                    task_idx += 1
            
            # 并行执行
            pool = self._get_pool()
            raw_results = pool.map(_simulate_single_action_worker, tasks)
            
            # 计算奖励并按候选分组
            all_results = [[] for _ in candidate_actions]
            for i, result in enumerate(raw_results):
                action_idx = task_mapping[i]
                reward = self._analyze_result_for_reward(result, last_state_s, valid_targets)
                
                # 额外检查首球犯规
                if result['success']:
                    first_contact = result['first_contact']
                    if first_contact and first_contact not in valid_targets:
                        reward = min(reward, -30.0)
                
                all_results[action_idx].append(reward)
            
            return all_results
            
        except Exception as e:
            print(f"[NewAgent] 并行MCTS失败: {e}")
            # 回退单进程
            all_results = []
            for idx, action in enumerate(candidate_actions):
                action_results = []
                for _ in range(sim_counts[idx]):
                    shot = self.simulate_action(balls, table, action)
                    if shot is None:
                        action_results.append(-500.0)
                    else:
                        is_legal, _ = self._check_first_contact(shot, valid_targets)
                        if not is_legal:
                            action_results.append(-30.0)
                        else:
                            action_results.append(analyze_shot_for_reward(shot, 
                                {bid: copy.deepcopy(b) for bid, b in balls.items()}, valid_targets))
                all_results.append(action_results)
            return all_results

    # ========== 8球犯规专项检测（多进程版）==========
    def _could_cause_8ball_foul(self, action, balls, table, valid_targets, tests=6):
        """专门检测动作是否可能导致8球犯规（支持并行）"""
        can_shoot_8 = ('8' in valid_targets)
        if can_shoot_8:
            return False
        
        last_state_s = {bid: ball.state.s for bid, ball in balls.items()}
        
        if self.use_multiprocess:
            try:
                balls_data = serialize_balls(balls)
                table_data = serialize_table(table)
                
                tasks = []
                base_seed = random.randint(0, 100000)
                for i in range(tests):
                    tasks.append((balls_data, table_data, action, self.sim_noise, base_seed + i))
                
                pool = self._get_pool()
                results = pool.map(_simulate_single_action_worker, tasks)
                
                for result in results:
                    if not result['success']:
                        continue
                    
                    new_pocketed = [bid for bid, info in result['balls_final'].items()
                                   if info['s'] == 4 and last_state_s.get(bid, 0) != 4]
                    
                    if '8' in new_pocketed:
                        return True
                    
                    if result['first_contact'] == '8':
                        return True
                
                return False
                
            except Exception:
                pass  # 回退单进程
        
        # 单进程模式
        for _ in range(tests):
            shot = self.simulate_action(balls, table, action)
            if shot is None:
                continue
            
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            if '8' in new_pocketed:
                return True
            
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if first_ball == '8':
                return True
        
        return False

    def _quick_first_contact_check(self, action, balls, table, valid_targets, tests=4):
        """快速检测动作是否可能导致首球犯规（支持并行）"""
        last_state_s = {bid: ball.state.s for bid, ball in balls.items()}
        
        if self.use_multiprocess:
            try:
                balls_data = serialize_balls(balls)
                table_data = serialize_table(table)
                
                tasks = []
                base_seed = random.randint(0, 100000)
                for i in range(tests):
                    tasks.append((balls_data, table_data, action, self.sim_noise, base_seed + i))
                
                pool = self._get_pool()
                results = pool.map(_simulate_single_action_worker, tasks)
                
                foul_count = 0
                for result in results:
                    if not result['success']:
                        foul_count += 1
                        continue
                    
                    first_contact = result['first_contact']
                    if first_contact is None or first_contact not in valid_targets:
                        foul_count += 1
                
                return foul_count >= tests // 2
                
            except Exception:
                pass
        
        # 单进程
        foul_count = 0
        for _ in range(tests):
            shot = self.simulate_action(balls, table, action)
            if shot is None:
                foul_count += 1
                continue
            
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                foul_count += 1
        
        return foul_count >= tests // 2

    # ========== 安全验证（多进程版）==========
    def _is_action_safe(self, action, balls, table, valid_targets, simulations=10):
        """验证动作安全性（支持并行）"""
        can_shoot_8 = ('8' in valid_targets)
        last_state_s = {bid: ball.state.s for bid, ball in balls.items()}
        
        if self.use_multiprocess:
            try:
                balls_data = serialize_balls(balls)
                table_data = serialize_table(table)
                
                tasks = []
                base_seed = random.randint(0, 100000)
                for i in range(simulations):
                    tasks.append((balls_data, table_data, action, self.sim_noise, base_seed + i))
                
                pool = self._get_pool()
                results = pool.map(_simulate_single_action_worker, tasks)
                
                fatal_count = 0
                eight_illegal_count = 0
                
                for result in results:
                    if not result['success']:
                        fatal_count += 1
                        continue
                    
                    new_pocketed = [bid for bid, info in result['balls_final'].items()
                                   if info['s'] == 4 and last_state_s.get(bid, 0) != 4]
                    
                    if 'cue' in new_pocketed:
                        fatal_count += 1
                        continue
                    
                    if '8' in new_pocketed and not can_shoot_8:
                        eight_illegal_count += 1
                        fatal_count += 3
                        continue
                    
                    first_contact = result['first_contact']
                    if first_contact is None or first_contact not in valid_targets:
                        if first_contact == '8' and not can_shoot_8:
                            eight_illegal_count += 1
                            fatal_count += 3
                        else:
                            fatal_count += 1
                
                if eight_illegal_count > 0:
                    return False
                return fatal_count <= 1
                
            except Exception:
                pass
        
        # 单进程
        fatal_count = 0
        first_contact_foul = 0
        cue_pocket_count = 0
        eight_illegal_count = 0
        
        for _ in range(simulations):
            shot = self.simulate_action(balls, table, action)
            if shot is None:
                fatal_count += 1
                continue
            
            new_pocketed = [bid for bid, b in shot.balls.items() 
                          if b.state.s == 4 and balls[bid].state.s != 4]
            
            if 'cue' in new_pocketed:
                cue_pocket_count += 1
                fatal_count += 1
                continue
            
            if '8' in new_pocketed and not can_shoot_8:
                eight_illegal_count += 1
                fatal_count += 3
                continue
            
            is_legal, first_ball = self._check_first_contact(shot, valid_targets)
            if not is_legal:
                first_contact_foul += 1
                if first_ball == '8' and not can_shoot_8:
                    eight_illegal_count += 1
                    fatal_count += 3
                else:
                    fatal_count += 1
                continue
        
        if eight_illegal_count > 0:
            return False
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
