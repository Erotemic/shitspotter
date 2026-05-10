package io.kitware.shitspotter.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.FailureType
import io.kitware.shitspotter.core.ModelRegistry

interface CameraSurface {
    @Composable
    fun Render(modifier: Modifier)
}

@Composable
fun AppRootTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = darkColorScheme()) {
        Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF000000)) {
            content()
        }
    }
}

@Composable
fun AppScreen(
    state: AppState,
    cameraSurface: CameraSurface,
    onSaveFailureCase: (FailureType, String?) -> Unit,
    onTogglePause: (() -> Unit)? = null,
    isPaused: Boolean = false,
) {
    AppRootTheme {
        Box(modifier = Modifier.fillMaxSize()) {
            cameraSurface.Render(Modifier.fillMaxSize())

            if (state.showOverlay) {
                DetectionOverlay(
                    detections = state.lastDetections,
                    frameWidth = state.lastFrameWidth,
                    frameHeight = state.lastFrameHeight,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(12.dp),
            ) {
                if (state.showFps) {
                    TelemetryHud(
                        telemetry = state.lastTelemetry,
                        detectionCount = state.lastDetections.size,
                    )
                }
                Spacer(Modifier.height(6.dp))
                ModelChipsRow(
                    activeId = state.activeModelId,
                    onSelect = { state.activeModelId = it },
                )
                state.lastError?.let { err ->
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "ERROR: $err",
                        color = Color(0xFFFF8080),
                        fontSize = 12.sp,
                        modifier = Modifier
                            .background(Color(0x88000000))
                            .padding(8.dp),
                    )
                }
            }

            ControlBar(
                state = state,
                onSaveFailureCase = onSaveFailureCase,
                onTogglePause = onTogglePause,
                isPaused = isPaused,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(12.dp),
            )
        }
    }
}

@Composable
private fun ControlBar(
    state: AppState,
    onSaveFailureCase: (FailureType, String?) -> Unit,
    onTogglePause: (() -> Unit)?,
    isPaused: Boolean,
    modifier: Modifier = Modifier,
) {
    var showFailureMenu by remember { mutableStateOf(false) }
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (showFailureMenu) {
            FailureTypePicker(
                onPick = { type ->
                    showFailureMenu = false
                    onSaveFailureCase(type, null)
                },
                onCancel = { showFailureMenu = false },
            )
            Spacer(Modifier.height(8.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { showFailureMenu = !showFailureMenu }) {
                Text(if (showFailureMenu) "Cancel" else "Save failure (${state.failureCasesSavedCount})")
            }
            onTogglePause?.let {
                Button(onClick = it) {
                    Text(if (isPaused) "Resume" else "Pause")
                }
            }
        }
    }
}

/**
 * Tiny chip-style model selector. The active model is bolded; tapping a
 * non-active chip writes the new id into [AppState.activeModelId]. The
 * actual backend swap is done lazily by the host (Android MainActivity
 * or desktop Main) the next time it picks up the registry id — this
 * composable does not own backend lifecycle.
 */
@Composable
private fun ModelChipsRow(
    activeId: String,
    onSelect: (String) -> Unit,
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        ModelRegistry.all.forEach { spec ->
            val active = spec.modelId == activeId
            Button(
                onClick = { if (!active) onSelect(spec.modelId) },
            ) {
                Text(
                    text = if (active) "● ${spec.displayName}" else spec.displayName,
                    color = Color.White,
                )
            }
        }
    }
}

@Composable
private fun FailureTypePicker(
    onPick: (FailureType) -> Unit,
    onCancel: () -> Unit,
) {
    Column(
        modifier = Modifier
            .background(Color(0xCC222222))
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        FailureType.values().forEach { ft ->
            Button(onClick = { onPick(ft) }, modifier = Modifier.width(220.dp)) {
                Text(ft.name)
            }
        }
        Button(onClick = onCancel, modifier = Modifier.width(220.dp)) {
            Text("cancel")
        }
    }
}

