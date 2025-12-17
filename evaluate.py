"""evaluate.py - Agent 评估脚本（增强版）

新增功能：
1. 详细犯规统计
2. 每局击球数记录
3. 胜负原因分析
4. 可视化统计图表

Notes:
- Default behavior matches the original script.
- Optional CLI args allow reproducible / smaller evaluations for iteration.
"""

import argparse
import json

from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate BasicAgent vs NewAgent")
    parser.add_argument("--games", type=int, default=100, help="number of games to play (default: 100)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="when provided, enables deterministic seeding via utils.set_random_seed",
    )
    args = parser.parse_args()

    # 设置随机种子（默认与原脚本一致：不固定随机性）
    if args.seed is None:
        set_random_seed(enable=False, seed=42)
    else:
        set_random_seed(enable=True, seed=int(args.seed))

    env = PoolEnv()

    # ========== 基础结果统计 ==========
    results = {
        'AGENT_A_WIN': 0,
        'AGENT_B_WIN': 0,
        'SAME': 0
    }

    # ========== 新增：详细犯规统计 ==========
    foul_stats = {
        'AGENT_A': {
            'cue_pocket': 0,      # 白球进袋
            'eight_illegal': 0,   # 误打黑8
            'first_foul': 0,      # 首球犯规
            'rail_foul': 0,       # 碰库犯规
            'no_hit': 0,          # 未击中任何球
            'total_shots': 0,     # 总击球数
        },
        'AGENT_B': {
            'cue_pocket': 0,
            'eight_illegal': 0,
            'first_foul': 0,
            'rail_foul': 0,
            'no_hit': 0,
            'total_shots': 0,
        }
    }

    # ========== 新增：每局详细记录 ==========
    game_logs = []

    n_games = int(args.games)
    agent_a, agent_b = BasicAgent(), NewAgent()
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

    print("=" * 60)
    print(f"开始对战评估：共 {n_games} 局")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"Agent A: {agent_a.__class__.__name__}")
    print(f"Agent B: {agent_b.__class__.__name__}")
    print("=" * 60)

    for i in range(n_games):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{n_games} 局比赛")
        print(f"{'='*60}")

        env.reset(target_ball=target_ball_choice[i % 4])
        player_class_a = players[i % 2].__class__.__name__
        player_class_b = players[(i + 1) % 2].__class__.__name__
        ball_type = target_ball_choice[i % 4]

        print(f"Player A: {player_class_a} ({ball_type})")
        print(f"Player B: {player_class_b}")

        # 本局统计
        game_log = {
            'game_id': i,
            'player_a_agent': player_class_a,
            'player_b_agent': player_class_b,
            'player_a_ball_type': ball_type,
            'shots': 0,
            'fouls': [],
            'winner': None
        }

        while True:
            player = env.get_curr_player()
            print(f"\n[第{env.hit_count}杆] Player {player} 击球")

            obs = env.get_observation(player)

            # 根据当前player选择对应agent
            if player == 'A':
                action = players[i % 2].decision(*obs)
                current_agent = 'AGENT_A' if (i % 2 == 0) else 'AGENT_B'
            else:
                action = players[(i + 1) % 2].decision(*obs)
                current_agent = 'AGENT_B' if (i % 2 == 0) else 'AGENT_A'

            # 统计击球数
            foul_stats[current_agent]['total_shots'] += 1

            step_info = env.take_shot(action)

            # ========== 新增：犯规统计 ==========
            foul_this_shot = []

            if step_info.get('WHITE_BALL_INTO_POCKET'):
                foul_stats[current_agent]['cue_pocket'] += 1
                foul_this_shot.append('白球进袋')
                print(f"   ❌ 犯规：白球进袋")

            if step_info.get('BLACK_BALL_INTO_POCKET'):
                # 检查是否是误打黑8（需要看是否获胜）
                done, info = env.get_done()
                if done and info['winner'] != player:
                    foul_stats[current_agent]['eight_illegal'] += 1
                    foul_this_shot.append('误打黑8')
                    print(f"   ❌ 犯规：误打黑8")

            if step_info.get('FOUL_FIRST_HIT'):
                foul_stats[current_agent]['first_foul'] += 1
                foul_this_shot.append('首球犯规')
                print(f"   ❌ 犯规：首球碰触对方球")

            if step_info.get('NO_POCKET_NO_RAIL'):
                foul_stats[current_agent]['rail_foul'] += 1
                foul_this_shot.append('碰库犯规')
                print(f"   ❌ 犯规：无进球且未碰库")

            if step_info.get('NO_HIT'):
                foul_stats[current_agent]['no_hit'] += 1
                foul_this_shot.append('未击中')
                print(f"   ❌ 犯规：白球未接触任何球")

            if foul_this_shot:
                game_log['fouls'].append({
                    'shot': env.hit_count,
                    'player': player,
                    'agent': current_agent,
                    'types': foul_this_shot
                })

            # ========== 进球提示 ==========
            if step_info.get('ME_INTO_POCKET'):
                print(f"   ✅ 进球：{step_info['ME_INTO_POCKET']}")

            if step_info.get('ENEMY_INTO_POCKET'):
                print(f"   ⚠️  对方球进袋：{step_info['ENEMY_INTO_POCKET']}")

            done, info = env.get_done()
            if done:
                game_log['shots'] = env.hit_count
                game_log['winner'] = info['winner']

                # 统计胜负
                if info['winner'] == 'SAME':
                    results['SAME'] += 1
                    print(f"\n🤝 平局！({env.hit_count}杆)")
                elif info['winner'] == 'A':
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                    winner_agent = ['AGENT_A', 'AGENT_B'][i % 2]
                    print(f"\n🏆 Player A 获胜 ({winner_agent})！({env.hit_count}杆)")
                else:
                    results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
                    winner_agent = ['AGENT_A', 'AGENT_B'][(i+1) % 2]
                    print(f"\n🏆 Player B 获胜 ({winner_agent})！({env.hit_count}杆)")

                game_logs.append(game_log)
                break

    # ========== 计算最终得分 ==========
    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

    # ========== 输出统计报告 ==========
    print("\n" + "=" * 60)
    print("📊 最终统计报告")
    print("=" * 60)

    print(f"\n【胜负结果】")
    print(f"  Agent A ({agent_a.__class__.__name__}):")
    print(f"    胜: {results['AGENT_A_WIN']} 局")
    print(f"    负: {results['AGENT_B_WIN']} 局")
    print(f"    平: {results['SAME']} 局")
    print(f"    得分: {results['AGENT_A_SCORE']:.1f}")
    print(f"    胜率: {results['AGENT_A_SCORE'] / n_games * 100:.1f}%")

    print(f"\n  Agent B ({agent_b.__class__.__name__}):")
    print(f"    胜: {results['AGENT_B_WIN']} 局")
    print(f"    负: {results['AGENT_A_WIN']} 局")
    print(f"    平: {results['SAME']} 局")
    print(f"    得分: {results['AGENT_B_SCORE']:.1f}")
    print(f"    胜率: {results['AGENT_B_SCORE'] / n_games * 100:.1f}%")

    # ========== 犯规统计 ==========
    print(f"\n【犯规统计】")
    for agent_name, stats in foul_stats.items():
        agent_class = agent_a.__class__.__name__ if agent_name == 'AGENT_A' else agent_b.__class__.__name__
        total_fouls = sum([
            stats['cue_pocket'],
            stats['eight_illegal'],
            stats['first_foul'],
            stats['rail_foul'],
            stats['no_hit']
        ])

        print(f"\n  {agent_name} ({agent_class}):")
        print(f"    总击球数: {stats['total_shots']}")
        print(f"    总犯规数: {total_fouls}")
        print(f"    犯规率: {total_fouls / stats['total_shots'] * 100:.1f}%" if stats['total_shots'] > 0 else "    犯规率: 0%")
        print(f"    ├─ 白球进袋: {stats['cue_pocket']}")
        print(f"    ├─ 误打黑8: {stats['eight_illegal']}")
        print(f"    ├─ 首球犯规: {stats['first_foul']}")
        print(f"    ├─ 碰库犯规: {stats['rail_foul']}")
        print(f"    └─ 未击中球: {stats['no_hit']}")

    # ========== 平均击球数统计 ==========
    total_shots = sum([log['shots'] for log in game_logs])
    avg_shots = total_shots / n_games if n_games > 0 else 0
    print(f"\n【效率统计】")
    print(f"  平均每局击球数: {avg_shots:.1f}")
    print(f"  最短对局: {min([log['shots'] for log in game_logs])} 杆")
    print(f"  最长对局: {max([log['shots'] for log in game_logs])} 杆")

    # ========== 保存详细日志 ==========
    try:
        with open('evaluation_log.json', 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'foul_stats': foul_stats,
                'game_logs': game_logs
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 详细日志已保存到: evaluation_log.json")
    except Exception as e:
        print(f"\n⚠️  日志保存失败: {e}")

    # ========== 性能评级 ==========
    win_rate = results['AGENT_B_SCORE'] / n_games
    print(f"\n【性能评级】")
    if win_rate >= 0.7:
        grade = "🏆 优秀"
    elif win_rate >= 0.6:
        grade = "⭐ 良好"
    elif win_rate >= 0.5:
        grade = "✅ 及格"
    elif win_rate >= 0.4:
        grade = "⚠️  较弱"
    else:
        grade = "❌ 不足"

    print(f"  Agent B 性能: {grade} (胜率 {win_rate*100:.1f}%)")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())