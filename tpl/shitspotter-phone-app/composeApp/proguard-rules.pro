# ShitSpotter v2 phone app — release ProGuard / R8 rules
#
# Goal: keep the release APK small + obfuscated, but never strip the
# bits that ONNX Runtime, kotlinx.serialization, or Compose need.

# kotlinx.serialization classes are accessed by reflection.
-keepattributes *Annotation*, InnerClasses
-keep,includedescriptorclasses class kotlinx.serialization.** { *; }
-keep,includedescriptorclasses class io.kitware.shitspotter.core.** { *; }

# Compose Multiplatform: keep Composable annotations + reflection paths.
-keepclassmembers class * {
    @androidx.compose.runtime.Composable *;
}

# CameraX uses reflection for some delegate paths.
-keep class androidx.camera.** { *; }

# ONNX Runtime native bindings.
-keep class ai.onnxruntime.** { *; }

# Kotlin coroutines + reflection for serialization.
-keepclassmembernames class kotlinx.coroutines.** {
    volatile <fields>;
}

# Strip Android system noise but keep our top-level entry.
-keep public class io.kitware.shitspotter.android.MainActivity {
    public protected *;
}
