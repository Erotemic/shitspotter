import java.time.LocalDate
import java.util.Properties
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
                implementation(libs.androidx.exifinterface)
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
    namespace = "io.github.erotemic.shitspotter"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.erotemic.shitspotter"
        minSdk = 24
        targetSdk = 35
        versionCode = providers.gradleProperty("APP_VERSION_CODE").getOrElse("1").toInt()
        versionName = providers.gradleProperty("APP_VERSION_NAME").getOrElse("0.1.0")

        // Pixel 5 is arm64-v8a; filtering shrinks the APK from ~80 MB
        // (4 ABIs of ONNX Runtime) to ~25 MB. Future agent: add x86_64
        // back if running in an emulator, or armeabi-v7a for older
        // phones.
        ndk {
            abiFilters += setOf("arm64-v8a")
        }
    }

    // Release signing: reads from local.properties (gitignored) or env vars.
    // Copy local.properties.example → local.properties and fill in values, or
    // set RELEASE_KEYSTORE_PATH / RELEASE_KEYSTORE_PASSWORD / RELEASE_KEY_ALIAS /
    // RELEASE_KEY_PASSWORD in the environment before building.
    signingConfigs {
        create("release") {
            val props = Properties()
            val localProps = rootProject.file("local.properties")
            if (localProps.exists()) props.load(localProps.inputStream())
            fun env(key: String, prop: String) =
                System.getenv(key) ?: props.getProperty(prop)
            val ksPath = env("RELEASE_KEYSTORE_PATH", "releaseKeystorePath")
            storeFile = ksPath?.let { file(it) }
            storePassword = env("RELEASE_KEYSTORE_PASSWORD", "releaseKeystorePassword") ?: ""
            keyAlias = env("RELEASE_KEY_ALIAS", "releaseKeyAlias") ?: ""
            keyPassword = env("RELEASE_KEY_PASSWORD", "releaseKeyPassword") ?: ""
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
        val gitCommit = providers.exec { commandLine("git", "rev-parse", "--short", "HEAD") }
            .standardOutput.asText.get().trim().ifEmpty { "unknown" }
        val buildDate = LocalDate.now().toString()
        debug {
            applicationIdSuffix = ".debug"
            isMinifyEnabled = false
            buildConfigField("String", "APP_GIT_COMMIT", "\"$gitCommit\"")
            buildConfigField("String", "APP_BUILD_DATE", "\"$buildDate\"")
        }
        release {
            val releaseSigning = signingConfigs.getByName("release")
            // Use release keystore if configured; fall back to debug for local dev.
            signingConfig = if (releaseSigning.storeFile != null) releaseSigning
                            else signingConfigs.getByName("debug")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            buildConfigField("String", "APP_GIT_COMMIT", "\"$gitCommit\"")
            buildConfigField("String", "APP_BUILD_DATE", "\"$buildDate\"")
        }
    }
}

compose.desktop {
    application {
        mainClass = "io.github.erotemic.shitspotter.desktop.MainKt"

        nativeDistributions {
            targetFormats(TargetFormat.Deb, TargetFormat.AppImage)
            packageName = "ScatSpotter"
            packageVersion = "0.1.0"
        }
    }
}
