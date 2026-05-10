import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.application)
    alias(libs.plugins.compose.multiplatform)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    androidTarget {
        compilations.all {
            kotlinOptions {
                jvmTarget = "17"
            }
        }
    }

    jvm("desktop") {
        compilations.all {
            kotlinOptions {
                jvmTarget = "17"
            }
        }
    }

    // iOS targets are declared so the KMP source set layout is correct,
    // but they only build on a macOS host with Xcode. From Linux these
    // targets show up as "configured but not buildable" in Gradle, which
    // is the right behaviour — `iosMain/` source set still gets type-
    // checked against commonMain.
    val isLinux = System.getProperty("os.name")?.lowercase()?.contains("linux") == true
    if (!isLinux || providers.gradleProperty("ssp.enableIosTargets").orNull == "true") {
        iosArm64()
        iosX64()
        iosSimulatorArm64()
    }

    targets.all {
        compilations.all {
            kotlinOptions {
                freeCompilerArgs += listOf("-Xexpect-actual-classes")
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(compose.runtime)
                implementation(compose.foundation)
                implementation(compose.material3)
                implementation(compose.ui)
                implementation(compose.components.resources)
                implementation(compose.components.uiToolingPreview)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.kotlinx.datetime)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val androidMain by getting {
            dependencies {
                implementation(libs.androidx.activity.compose)
                implementation(libs.androidx.core.ktx)
                implementation(libs.androidx.lifecycle.runtime.ktx)
                implementation(libs.androidx.lifecycle.viewmodel.compose)
                implementation(libs.androidx.camera.core)
                implementation(libs.androidx.camera.camera2)
                implementation(libs.androidx.camera.lifecycle)
                implementation(libs.androidx.camera.view)
                implementation(libs.kotlinx.coroutines.android)
                implementation(libs.onnxruntime.android)
            }
        }
        val desktopMain by getting {
            dependencies {
                implementation(compose.desktop.currentOs)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.onnxruntime.jvm)
            }
        }
    }
}

android {
    namespace = "io.kitware.shitspotter"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.kitware.shitspotter"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        // Pixel 5 is arm64-v8a; filtering shrinks the APK from ~80 MB
        // (4 ABIs of ONNX Runtime) to ~25 MB. Future agent: add x86_64
        // back if running in an emulator, or armeabi-v7a for older
        // phones.
        ndk {
            abiFilters += setOf("arm64-v8a")
        }
    }

    buildFeatures {
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "META-INF/DEPENDENCIES"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].apply {
        manifest.srcFile("src/androidMain/AndroidManifest.xml")
        res.srcDirs("src/androidMain/res")
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            buildConfigField(
                "String",
                "APP_GIT_COMMIT",
                "\"${providers.exec { commandLine("git", "rev-parse", "--short", "HEAD") }
                    .standardOutput.asText.get().trim().ifEmpty { "unknown" }}\""
            )
        }
        release {
            // Note: debuggable signing config is reused — release builds
            // are not yet wired for Play Store signing. The user adds a
            // real signingConfig + keystore before publishing.
            signingConfig = signingConfigs.getByName("debug")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            buildConfigField(
                "String",
                "APP_GIT_COMMIT",
                "\"${providers.exec { commandLine("git", "rev-parse", "--short", "HEAD") }
                    .standardOutput.asText.get().trim().ifEmpty { "unknown" }}\""
            )
        }
    }
}

compose.desktop {
    application {
        mainClass = "io.kitware.shitspotter.desktop.MainKt"

        nativeDistributions {
            targetFormats(TargetFormat.Deb, TargetFormat.AppImage)
            packageName = "ShitSpotter"
            packageVersion = "0.1.0"
        }
    }
}
