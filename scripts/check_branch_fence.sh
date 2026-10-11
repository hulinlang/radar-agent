#!/usr/bin/env bash
# check_branch_fence.sh -- 强制 docs/35 5.3 的路由分支不变量。
#
# 不变量：`route/*` 分支只允许修改**自己的路由专属路径**；共享路径在路由分支上只读。
#
# 为什么重要：只要这条不变量成立，`git merge main` 进路由分支**永远不会冲突**
# （路由分支从没碰过 main 也在改的文件），于是共享层的一次修改只需"merge 两次"
# 就同步到两条路线，而不是"手工改两遍"。破坏一次，之后每次同步都开始冲突，
# 分支策略会迅速退化成维护两份代码。
#
# 用法：  bash scripts/check_branch_fence.sh
# 退出码：0 = 干净（或当前不是 route/* 分支）；1 = 违规
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

branch=$(git rev-parse --abbrev-ref HEAD)
case "$branch" in
  route/radarinfer)
    allow=(
      'cpp/*'
      'src/engines/react_radarinfer.py'
      'src/perf_model.py'
      'configs/perf_model.yaml'
      'scripts/p11_perf_model.py'
      'scripts/p9_compare_radarinfer.py'
      'scripts/p10_wsl_radar_serve_start.sh'
      'reports/P11_Roofline性能模型.md'
      'reports/p11_roofline.png'
      'reports/P7_端到端评测_p9radarinfer2.md'
      'docs/2[0-9]_*'
      'docs/3[0-6]_*'
    ) ;;
  route/vllm)
    allow=(
      'scripts/p9_wsl_vllm_*'
      'docs/18_*'
    ) ;;
  *)
    echo "branch '$branch' 不是 route/* 分支 —— fence 不适用"
    exit 0 ;;
esac

base=$(git merge-base main HEAD) || { echo "找不到与 main 的 merge base"; exit 1; }
echo "branch : $branch"
echo "base   : $(git rev-parse --short "$base")  (= merge-base with main)"
echo "changed:"
mapfile -t changed < <(git diff --name-only "$base" HEAD | tr -d '\r')
viol=()
for f in "${changed[@]}"; do
  okf=0
  for p in "${allow[@]}"; do
    case "$f" in $p) okf=1; break ;; esac
  done
  if [ "$okf" -eq 1 ]; then
    echo "  ok        $f"
  else
    echo "  VIOLATION $f"
    viol+=("$f")
  fi
done
echo
if [ "${#viol[@]}" -eq 0 ]; then
  echo "FENCE OK: ${#changed[@]} 个改动文件，全部属于本路由分支。"
  exit 0
fi
echo "FENCE VIOLATION: 在 '$branch' 上改动了 ${#viol[@]} 个共享路径："
printf '  %s\n' "${viol[@]}"
echo
echo "共享路径在路由分支上只读（docs/35 5.3）。"
echo "把这个改动放回 'main'，再 merge 回来。"
exit 1
