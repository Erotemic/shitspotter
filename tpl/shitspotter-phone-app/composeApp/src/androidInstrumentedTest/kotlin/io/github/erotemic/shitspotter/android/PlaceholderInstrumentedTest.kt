package io.github.erotemic.shitspotter.android

/**
 * NOT YET ENABLED. This file exists so that the
 * `composeApp/src/androidInstrumentedTest/` source set is present in the
 * KMP source tree, ready for an agent to add a real test against an
 * emulator or connected device.
 *
 * To turn it on:
 *
 * 1. Add the AGP `androidTestImplementation` dependencies to
 *    `composeApp/build.gradle.kts`:
 *
 *    ```kotlin
 *    androidTarget {
 *        // …existing config…
 *        instrumentedTestVariant.sourceSetTree.set(KotlinSourceSetTree.test)
 *    }
 *    sourceSets {
 *        val androidInstrumentedTest by getting {
 *            dependencies {
 *                implementation("androidx.test:runner:1.6.2")
 *                implementation("androidx.test:rules:1.6.1")
 *                implementation("androidx.test.ext:junit:1.2.1")
 *                implementation("androidx.compose.ui:ui-test-junit4-android:1.7.4")
 *            }
 *        }
 *    }
 *    ```
 *
 * 2. Add a real test file (delete this placeholder):
 *
 *    ```kotlin
 *    @RunWith(AndroidJUnit4::class)
 *    class CameraAnalysisLoopTest {
 *        @Test fun keep_only_latest_drops_stale_frames() { ... }
 *    }
 *    ```
 *
 * 3. Run the tests against a connected device or emulator:
 *
 *    ```bash
 *    ./gradlew :composeApp:connectedAndroidTest
 *    ```
 *
 * The VM has no USB passthrough, so this is a workstation- or CI-only
 * task — the same sideload path the user already uses for the APK.
 */
@Suppress("unused")
internal object PlaceholderInstrumentedTest
