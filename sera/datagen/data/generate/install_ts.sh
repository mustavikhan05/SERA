#!/bin/bash
# install_ts.sh — TypeScript repository setup inside Docker
#
# This script is the TypeScript equivalent of install.sh (Python/conda).
# It is executed during Docker image build to prepare a TypeScript repo
# for test execution and bug introduction/fixing by SWE-agent.
#
# Expected environment:
#   - node:20-slim base image with Node.js 20.x pre-installed
#   - pnpm installed globally (npm install -g pnpm)
#   - Jelly installed globally (npm install -g @cs-au-dk/jelly)
#   - Repo source code copied into /repo (WORKDIR)
#
# Optional env vars:
#   SERA_PKG_MANAGER  — force a specific package manager (pnpm|yarn|npm)
#   SERA_SKIP_TSC     — set to "1" to skip TypeScript compilation check
#   SERA_SKIP_TESTS   — set to "1" to skip test framework verification

set -euo pipefail

echo "=========================================="
echo " SERA TypeScript Install Script"
echo "=========================================="
echo ""

###############################################################################
# 1. Detect package manager from lockfiles (or use override)
###############################################################################

if [ -n "${SERA_PKG_MANAGER:-}" ]; then
    PKG_MANAGER="$SERA_PKG_MANAGER"
    echo "> Using package manager override: $PKG_MANAGER"
elif [ -f "pnpm-lock.yaml" ]; then
    PKG_MANAGER="pnpm"
    echo "> Detected pnpm (pnpm-lock.yaml found)"
elif [ -f "yarn.lock" ]; then
    PKG_MANAGER="yarn"
    echo "> Detected yarn (yarn.lock found)"
elif [ -f "package-lock.json" ]; then
    PKG_MANAGER="npm"
    echo "> Detected npm (package-lock.json found)"
else
    PKG_MANAGER="npm"
    echo "> No lockfile found, defaulting to npm"
fi

###############################################################################
# 2. Enable corepack if available (for pnpm/yarn version management)
###############################################################################

if command -v corepack &>/dev/null; then
    echo "> Enabling corepack for package manager version management..."
    corepack enable || echo "WARN: corepack enable failed (non-fatal)"
fi

###############################################################################
# 3. Install dependencies
###############################################################################

echo ""
echo "> Installing dependencies with $PKG_MANAGER..."

case "$PKG_MANAGER" in
    pnpm)
        # --frozen-lockfile ensures reproducible installs; fall back to regular
        # install if the lockfile is out of date (common at older commits)
        if pnpm install --frozen-lockfile 2>/dev/null; then
            echo "> pnpm install --frozen-lockfile succeeded"
        else
            echo "WARN: --frozen-lockfile failed, retrying with pnpm install (no lockfile enforcement)"
            pnpm install --no-frozen-lockfile
        fi
        ;;
    yarn)
        if yarn install --frozen-lockfile 2>/dev/null; then
            echo "> yarn install --frozen-lockfile succeeded"
        else
            echo "WARN: --frozen-lockfile failed, retrying with yarn install"
            yarn install
        fi
        ;;
    npm)
        if [ -f "package-lock.json" ]; then
            if npm ci 2>/dev/null; then
                echo "> npm ci succeeded"
            else
                echo "WARN: npm ci failed, retrying with npm install"
                npm install
            fi
        else
            npm install
        fi
        ;;
    *)
        echo "ERROR: Unknown package manager: $PKG_MANAGER"
        exit 1
        ;;
esac

echo "> Dependencies installed successfully"

###############################################################################
# 4. Verify TypeScript compilation
###############################################################################

echo ""
if [ "${SERA_SKIP_TSC:-0}" = "1" ]; then
    echo "> Skipping TypeScript compilation check (SERA_SKIP_TSC=1)"
else
    echo "> Verifying TypeScript compilation (npx tsc --noEmit)..."
    if npx tsc --noEmit 2>&1; then
        echo "> TypeScript compilation: OK"
    else
        # Many repos have tsc errors that are expected (e.g., strictNullChecks
        # not fully enabled, or errors in test files). We warn but don't fail.
        echo "WARN: tsc --noEmit reported errors (may be expected for this repo)"
    fi
fi

###############################################################################
# 5. Detect test framework and verify tests can run
###############################################################################

echo ""
if [ "${SERA_SKIP_TESTS:-0}" = "1" ]; then
    echo "> Skipping test verification (SERA_SKIP_TESTS=1)"
else
    echo "> Detecting test framework..."

    # Check package.json for test framework
    TEST_FRAMEWORK=""

    if [ -f "package.json" ]; then
        # Check devDependencies and dependencies for vitest/jest
        if grep -q '"vitest"' package.json 2>/dev/null; then
            TEST_FRAMEWORK="vitest"
        elif grep -q '"jest"' package.json 2>/dev/null; then
            TEST_FRAMEWORK="jest"
        fi

        # Also check the "test" script in package.json
        if [ -z "$TEST_FRAMEWORK" ]; then
            TEST_SCRIPT=$(node -e "try{console.log(require('./package.json').scripts?.test||'')}catch(e){console.log('')}" 2>/dev/null || echo "")
            if echo "$TEST_SCRIPT" | grep -q "vitest"; then
                TEST_FRAMEWORK="vitest"
            elif echo "$TEST_SCRIPT" | grep -q "jest"; then
                TEST_FRAMEWORK="jest"
            elif echo "$TEST_SCRIPT" | grep -q "mocha"; then
                TEST_FRAMEWORK="mocha"
            fi
        fi
    fi

    if [ -n "$TEST_FRAMEWORK" ]; then
        echo "> Detected test framework: $TEST_FRAMEWORK"
        echo "> Running smoke test (first 50 lines of output)..."
        case "$TEST_FRAMEWORK" in
            vitest)
                npx vitest run --reporter=verbose 2>&1 | head -50 || echo "WARN: vitest run had failures (may be expected)"
                ;;
            jest)
                npx jest --passWithNoTests 2>&1 | head -50 || echo "WARN: jest had failures (may be expected)"
                ;;
            mocha)
                npx mocha 2>&1 | head -50 || echo "WARN: mocha had failures (may be expected)"
                ;;
        esac
    else
        echo "> No test framework detected in package.json"
        # Try running the npm test script if it exists
        if [ -f "package.json" ] && node -e "process.exit(require('./package.json').scripts?.test ? 0 : 1)" 2>/dev/null; then
            echo "> Found 'test' script in package.json, running smoke test..."
            case "$PKG_MANAGER" in
                pnpm) pnpm test 2>&1 | head -50 || echo "WARN: pnpm test had failures" ;;
                yarn) yarn test 2>&1 | head -50 || echo "WARN: yarn test had failures" ;;
                npm)  npm test 2>&1 | head -50 || echo "WARN: npm test had failures" ;;
            esac
        else
            echo "> No test script found; skipping test verification"
        fi
    fi
fi

###############################################################################
# 6. Summary
###############################################################################

echo ""
echo "=========================================="
echo " SERA TypeScript Install: COMPLETE"
echo "=========================================="
echo " Package manager: $PKG_MANAGER"
echo " Node.js version: $(node --version)"
echo " npm version:     $(npm --version)"
if command -v pnpm &>/dev/null; then
    echo " pnpm version:    $(pnpm --version)"
fi
if command -v tsc &>/dev/null || npx tsc --version &>/dev/null 2>&1; then
    echo " tsc version:     $(npx tsc --version 2>/dev/null || echo 'N/A')"
fi
echo "=========================================="
