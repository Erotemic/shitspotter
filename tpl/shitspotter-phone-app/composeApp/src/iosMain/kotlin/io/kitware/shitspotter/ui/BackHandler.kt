package io.kitware.shitspotter.ui

import androidx.compose.runtime.Composable

@Composable
actual fun PlatformBackHandler(enabled: Boolean, onBack: () -> Unit) {
    // iOS back gesture is handled by the navigation host; no-op here.
}
