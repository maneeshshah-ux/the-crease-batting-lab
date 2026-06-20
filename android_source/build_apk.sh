#!/bin/bash
#
# the CREASE Batting Lab — APK Builder
# Builds an Android APK from source using Android SDK tools.
#

set -e

# ---------- Configuration ----------
JAVA_HOME="${JAVA_HOME:-/tmp/jdk-17.0.12+7/Contents/Home}"
ANDROID_HOME="${ANDROID_HOME:-/tmp/android-sdk}"
BUILD_TOOLS="$ANDROID_HOME/build-tools/34.0.0"
PLATFORM="$ANDROID_HOME/platforms/android-34"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="the_CREASE"
PACKAGE="com.crease.battinglab"

export PATH="$JAVA_HOME/bin:$BUILD_TOOLS:$PATH"
export ANDROID_HOME

echo "========================================"
echo "  the CREASE Batting Lab — APK Builder"
echo "========================================"
echo "Java:      $(java -version 2>&1 | head -1)"
echo "AAPT:      $(which aapt)"
echo "D8:        $(which d8)"
echo "APKSigner: $(which apksigner)"
echo ""

# Clean build dir
BUILD_DIR="$PROJECT_DIR/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/classes" "$BUILD_DIR/dex" "$BUILD_DIR/apk"

echo "[1/6] Compiling resources with aapt..."
aapt package -f \
    -M "$PROJECT_DIR/app/src/main/AndroidManifest.xml" \
    -S "$PROJECT_DIR/app/src/main/res" \
    -I "$PLATFORM/android.jar" \
    -m \
    -F "$BUILD_DIR/apk/resources.apk" \
    -J "$BUILD_DIR/source"

echo "[2/6] Compiling Java source..."
javac -source 17 -target 17 \
    -classpath "$PLATFORM/android.jar" \
    -d "$BUILD_DIR/classes" \
    "$PROJECT_DIR/app/src/main/java/com/crease/battinglab/MainActivity.java"

echo "[3/6] Converting to DEX..."
d8 \
    --lib "$PLATFORM/android.jar" \
    --output "$BUILD_DIR/dex" \
    "$BUILD_DIR/classes/com/crease/battinglab/MainActivity.class"

echo "[4/6] Packaging APK..."
# First, unpack resources
cd "$BUILD_DIR/apk"
unzip -qo resources.apk -d apk_content/

# Create unsigned APK
aapt package -f \
    -M "$PROJECT_DIR/app/src/main/AndroidManifest.xml" \
    -S "$PROJECT_DIR/app/src/main/res" \
    -I "$PLATFORM/android.jar" \
    -F "$BUILD_DIR/apk/unsigned.apk" \
    --include-assets

# Add DEX files to APK
cd "$BUILD_DIR/dex"
aapt add "$BUILD_DIR/apk/unsigned.apk" classes.dex

# Add native libs if any
cd "$BUILD_DIR"

echo "[5/6] Signing APK..."
# Generate debug keystore
KEYSTORE="$BUILD_DIR/debug.keystore"
KEYPASS="android"
if [ ! -f "$KEYSTORE" ]; then
    keytool -genkey -v -keystore "$KEYSTORE" \
        -alias crease_debug \
        -keyalg RSA -keysize 2048 -validity 10000 \
        -storepass "$KEYPASS" -keypass "$KEYPASS" \
        -dname "CN=the CREASE, OU=Dev, O=CREASE, L=Cricket, ST=CA, C=US" 2>/dev/null
fi

# Zipalign
zipalign -v -p 4 "$BUILD_DIR/apk/unsigned.apk" "$BUILD_DIR/apk/aligned.apk"

# Sign
apksigner sign \
    --ks "$KEYSTORE" \
    --ks-pass "pass:$KEYPASS" \
    --ks-key-alias crease_debug \
    --out "$PROJECT_DIR/${APP_NAME}_v1.0.0.apk" \
    "$BUILD_DIR/apk/aligned.apk"

echo ""
echo "========================================"
echo "  ✅ APK Built Successfully!"
echo "========================================"
echo "  File: $PROJECT_DIR/${APP_NAME}_v1.0.0.apk"
echo "  Size: $(ls -lh "$PROJECT_DIR/${APP_NAME}_v1.0.0.apk" | awk '{print $5}')"
echo ""
echo "  Install on phone:"
echo "    adb install ${APP_NAME}_v1.0.0.apk"
echo "  Or simply transfer the APK to your phone and tap to install."
echo "========================================"
