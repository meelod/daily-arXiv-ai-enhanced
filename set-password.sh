#!/bin/bash
# Interactive password setup — independent of GitHub Actions.
#
# Run this once locally to bake a SHA-256 password hash into js/auth-config.js,
# then commit and push the result. The login gate will work for anyone visiting
# the site, regardless of whether the daily workflow has run.
#
# Usage:
#   bash set-password.sh         (prompts for password, hides input)
#   bash set-password.sh --clear (disables the gate — site becomes public)

set -e

CONFIG_FILE="js/auth-config.js"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found. Run this from the repo root." >&2
    exit 1
fi

if [ "$1" = "--clear" ]; then
    PASSWORD_HASH="DISABLED_NO_PASSWORD_SET_IN_SECRETS"
    echo "Disabling password gate (site will be public)..."
else
    echo -n "Enter password (input hidden): "
    stty -echo
    read PASSWORD
    stty echo
    echo ""

    if [ -z "$PASSWORD" ]; then
        echo "Error: empty password." >&2
        exit 1
    fi

    if [ ${#PASSWORD} -lt 8 ]; then
        echo "Warning: password is shorter than 8 characters. The hash is in a public JS file —"
        echo "short passwords can be brute-forced. Recommended: 16+ chars or a passphrase."
        echo -n "Continue anyway? [y/N] "
        read CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            echo "Aborted."
            exit 1
        fi
    fi

    echo -n "Confirm password (input hidden): "
    stty -echo
    read PASSWORD_CONFIRM
    stty echo
    echo ""

    if [ "$PASSWORD" != "$PASSWORD_CONFIRM" ]; then
        echo "Error: passwords don't match." >&2
        exit 1
    fi

    if command -v openssl >/dev/null 2>&1; then
        # awk '{print $NF}' picks the last field — works whether openssl prefixes
        # output with "(stdin)= " (Linux/old) or just emits the hex (macOS).
        PASSWORD_HASH=$(printf '%s' "$PASSWORD" | openssl dgst -sha256 -hex | awk '{print $NF}')
    elif command -v shasum >/dev/null 2>&1; then
        PASSWORD_HASH=$(printf '%s' "$PASSWORD" | shasum -a 256 | awk '{print $1}')
    else
        echo "Error: neither openssl nor shasum found. Install one of them." >&2
        exit 1
    fi

    if [ ${#PASSWORD_HASH} -ne 64 ]; then
        echo "Error: hash extraction failed (got ${#PASSWORD_HASH} chars, expected 64)." >&2
        exit 1
    fi
fi

# Replace whatever's currently between the quotes for passwordHash.
# Works on macOS (BSD sed needs `-i ''`) and Linux (GNU sed accepts `-i`).
if sed --version >/dev/null 2>&1; then
    sed -i -E "s|passwordHash: '[^']*'|passwordHash: '$PASSWORD_HASH'|" "$CONFIG_FILE"
else
    sed -i '' -E "s|passwordHash: '[^']*'|passwordHash: '$PASSWORD_HASH'|" "$CONFIG_FILE"
fi

echo ""
echo "Updated $CONFIG_FILE"
echo "  hash preview: ${PASSWORD_HASH:0:12}...${PASSWORD_HASH: -4}"
echo ""
echo "Next steps:"
echo "  git add $CONFIG_FILE"
echo "  git commit -m 'set login password'"
echo "  git push origin main"
echo ""
echo "After ~1-2 min the live site will require this password to view."
