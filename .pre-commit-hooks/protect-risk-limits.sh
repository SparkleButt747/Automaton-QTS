#!/usr/bin/env bash
# ==============================================================================
# protect-risk-limits.sh
#
# Pre-commit hook that rejects modifications to config/risk_limits.json
# unless the environment variable QTS_FORCE_RISK_LIMITS_CHANGE=1 is set,
# or the commit message contains the literal string [force-risk-limits].
#
# This protects the immutable risk controls from accidental modification.
# To override (e.g., for a deliberate, reviewed risk limit change):
#
#   QTS_FORCE_RISK_LIMITS_CHANGE=1 git commit -m "feat: adjust max drawdown"
#
# Or add [force-risk-limits] to your commit message:
#
#   git commit -m "risk: lower max_position_size_pct to 3% [force-risk-limits]"
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RESET='\033[0m'

# Check if the staged changes include risk_limits.json
if ! git diff --cached --name-only | grep -q "config/risk_limits\.json"; then
    # Not staged — nothing to do
    exit 0
fi

echo ""
echo -e "${YELLOW}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${YELLOW}║         RISK LIMITS MODIFICATION DETECTED                ║${RESET}"
echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo "  config/risk_limits.json contains IMMUTABLE risk controls."
echo "  Modifying these values can expose the system to catastrophic loss."
echo ""
echo "  Staged diff:"
echo ""
git diff --cached -- config/risk_limits.json | head -40
echo ""

# ── Override: environment variable ──────────────────────────────────────────
if [ "${QTS_FORCE_RISK_LIMITS_CHANGE:-0}" = "1" ]; then
    echo -e "${GREEN}Override active: QTS_FORCE_RISK_LIMITS_CHANGE=1${RESET}"
    echo -e "${GREEN}Proceeding with risk limits change. Ensure this has been reviewed.${RESET}"
    echo ""
    exit 0
fi

# ── Override: commit message keyword ────────────────────────────────────────
COMMIT_MSG_FILE="${GIT_DIR:-$(git rev-parse --git-dir)}/COMMIT_EDITMSG"
if [ -f "$COMMIT_MSG_FILE" ] && grep -q "\[force-risk-limits\]" "$COMMIT_MSG_FILE"; then
    echo -e "${GREEN}Override found in commit message: [force-risk-limits]${RESET}"
    echo -e "${GREEN}Proceeding with risk limits change. Ensure this has been reviewed.${RESET}"
    echo ""
    exit 0
fi

# ── Reject ───────────────────────────────────────────────────────────────────
echo -e "${RED}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${RED}║                    COMMIT REJECTED                       ║${RESET}"
echo -e "${RED}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo "  To override, use ONE of the following methods:"
echo ""
echo "  1. Set environment variable:"
echo "     QTS_FORCE_RISK_LIMITS_CHANGE=1 git commit -m '...'"
echo ""
echo "  2. Include keyword in commit message:"
echo "     git commit -m 'risk: your message here [force-risk-limits]'"
echo ""
echo "  Any change to risk limits MUST be:"
echo "    - Reviewed by at least one other team member"
echo "    - Documented with rationale in the commit message"
echo "    - Tested in simulation before production deployment"
echo ""
exit 1
