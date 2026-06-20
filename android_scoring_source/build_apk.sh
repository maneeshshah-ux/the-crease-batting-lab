#!/bin/bash
#
# the CREASE Scoring — APK Builder
# Builds an Android APK from source using Android SDK tools.
#

set -e

# ---------- Configuration ----------
JAVA_HOME="${JAVA_HOME:-/tmp/jdk-17.0.12+7/Contents/Home}"
ANDROID_HOME="${ANDROID_HOME:-/tmp/android-sdk}"
BUILD_TOOLS="$ANDROID_HOME/build-tools/34.0.0"
PLATFORM="$ANDROID_HOME/platforms/android-34"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="the_CREASE_Scoring"
APP_VERSION="v1.0.0"
PACKAGE="com.crease.scoring"

export PATH="$JAVA_HOME/bin:$BUILD_TOOLS:$PATH"
export ANDROID_HOME

echo "========================================"
echo "  the CREASE Scoring — APK Builder"
echo "========================================"
echo "Java:      $(java -version 2>&1 | head -1)"
echo "AAPT:      $(which aapt)"
echo "D8:        $(which d8)"
echo "APKSigner: $(which apksigner)"
echo ""

# Clean build dir
BUILD_DIR="$PROJECT_DIR/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/classes" "$BUILD_DIR/dex" "$BUILD_DIR/apk" "$BUILD_DIR/source"

echo "[1/6] Compiling resources with aapt..."
aapt package -f \
    -M "$PROJECT_DIR/app/src/main/AndroidManifest.xml" \
    -S "$PROJECT_DIR/app/src/main/res" \
    -I "$PLATFORM/android.jar" \
    -m \
    -F "$BUILD_DIR/apk/resources.apk" \
    -J "$BUILD_DIR/source"

echo "[2/6] Compiling Java source..."
# Compile generated R.java first, then MainActivity
javac -source 17 -target 17 \
    -classpath "$PLATFORM/android.jar" \
    -d "$BUILD_DIR/classes" \
    "$BUILD_DIR/source/com/crease/scoring/R.java"

javac -source 17 -target 17 \
    -classpath "$PLATFORM/android.jar:$BUILD_DIR/classes" \
    -d "$BUILD_DIR/classes" \
    "$PROJECT_DIR/app/src/main/java/com/crease/scoring/MainActivity.java"

echo "[3/6] Converting to DEX..."
# Collect all .class files using a temp file to handle spaces in paths
find "$BUILD_DIR/classes" -name "*.class" > "$BUILD_DIR/classes_list.txt"
d8 \
    --lib "$PLATFORM/android.jar" \
    --output "$BUILD_DIR/dex" \
    @"$BUILD_DIR/classes_list.txt"

echo "[4/6] Packaging APK..."
# Create unsigned APK using aapt directly
aapt package -f \
    -M "$PROJECT_DIR/app/src/main/AndroidManifest.xml" \
    -S "$PROJECT_DIR/app/src/main/res" \
    -A "$PROJECT_DIR/app/src/main/assets" \
    -I "$PLATFORM/android.jar" \
    -F "$BUILD_DIR/apk/unsigned.apk" \
    -v

# Add DEX files to APK
cd "$BUILD_DIR"
aapt add "$BUILD_DIR/apk/unsigned.apk" \
    "$BUILD_DIR/dex/classes.dex"

# Add AndroidManifest
aapt add "$BUILD_DIR/apk/unsigned.apk" \
    "$PROJECT_DIR/app/src/main/AndroidManifest.xml"

echo "[5/6] Signing APK..."
# Generate debug keystore
KEYSTORE="$BUILD_DIR/debug.keystore"
KEYPASS="android"
if [ ! -f "$KEYSTORE" ]; then
    keytool -genkey -v -keystore "$KEYSTORE" \
        -alias crease_scoring_debug \
        -keyalg RSA -keysize 2048 -validity 10000 \
        -storepass "$KEYPASS" -keypass "$KEYPASS" \
        -dname "CN=the CREASE Scoring, OU=Dev, O=CREASE, L=Cricket, ST=CA, C=US" 2>/dev/null
fi

# Zipalign
zipalign -v -p 4 "$BUILD_DIR/apk/unsigned.apk" "$BUILD_DIR/apk/aligned.apk"

# Sign
apksigner sign \
    --ks "$KEYSTORE" \
    --ks-pass "pass:$KEYPASS" \
    --ks-key-alias crease_scoring_debug \
    --out "$PROJECT_DIR/${APP_NAME}_${APP_VERSION}.apk" \
    "$BUILD_DIR/apk/aligned.apk"

echo ""
echo "========================================"
echo "  ✅ APK Built Successfully!"
echo "========================================"
echo "  File: $PROJECT_DIR/${APP_NAME}_${APP_VERSION}.apk"
echo "  Size: $(ls -lh "$PROJECT_DIR/${APP_NAME}_${APP_VERSION}.apk" | awk '{print $5}')"
echo ""
echo "  Install on phone:"
echo "    adb install ${APP_NAME}_${APP_VERSION}.apk"
echo "  Or simply transfer the APK to your phone and tap to install."
echo "========================================"
