package io.kitware.shitspotter.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.FailureType
import io.kitware.shitspotter.core.ModelRegistry
import io.kitware.shitspotter.core.ModelSpec

interface CameraSurface {
    /** How this surface scales its preview. The DetectionOverlay uses
     *  this to keep boxes aligned with what the user actually sees on
     *  screen. */
    val overlayScaleMode: OverlayScaleMode get() = OverlayScaleMode.FILL_CENTER

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
    canSaveFailureCase: Boolean = true,
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
                    scaleMode = cameraSurface.overlayScaleMode,
                )
            }

            // Top control stack — uses systemBarsPadding so it sits below
            // the status bar / camera cutout on Pixel 5 portrait.
            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .systemBarsPadding()
                    .padding(8.dp)
                    // Cap height so this stack can't push the bottom
                    // controls off-screen on a short phone; users scroll
                    // it if their model labels are long.
                    .heightIn(max = 360.dp)
                    .verticalScroll(rememberScrollState()),
            ) {
                if (state.showFps) {
                    TelemetryHud(
                        telemetry = state.lastTelemetry,
                        detectionCount = state.lastDetections.size,
                    )
                }
                Spacer(Modifier.height(6.dp))
                val activeBackendName = state.lastTelemetry?.runtimeBackend
                val activeIsStubFallback =
                    state.activeModelId != "stub-fake-detector" &&
                        activeBackendName != null &&
                        activeBackendName.startsWith("stub-")
                ModelSelectorButton(
                    activeId = state.activeModelId,
                    activeIsStubFallback = activeIsStubFallback,
                    onSelect = { state.activeModelId = it },
                )
                if (activeIsStubFallback) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "⚠ model file not on device — using stub. " +
                            "adb push the .onnx into " +
                            "<external-files>/models/ and retap.",
                        color = Color(0xFFFFCC66),
                        fontSize = 11.sp,
                        modifier = Modifier
                            .background(Color(0x88000000))
                            .padding(8.dp),
                    )
                }
                Spacer(Modifier.height(4.dp))
                ScoreThresholdControl(state)
                Spacer(Modifier.height(4.dp))
                ToggleRow(state)
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

            // Bottom control bar — also system-bars-padded so the
            // navigation pill on Pixel 5 doesn't eat the buttons. The
            // failure-type picker is a Dialog overlay, so it does not
            // contribute to this bar's height any more.
            ControlBar(
                state = state,
                onSaveFailureCase = onSaveFailureCase,
                onTogglePause = onTogglePause,
                isPaused = isPaused,
                canSaveFailureCase = canSaveFailureCase,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .systemBarsPadding()
                    .padding(horizontal = 12.dp, vertical = 12.dp),
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
    canSaveFailureCase: Boolean,
    modifier: Modifier = Modifier,
) {
    var showFailureDialog by remember { mutableStateOf(false) }
    if (showFailureDialog) {
        FailureTypeDialog(
            onPick = { type, note ->
                showFailureDialog = false
                onSaveFailureCase(type, note)
            },
            onDismiss = { showFailureDialog = false },
        )
    }
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
    ) {
        Button(
            onClick = { showFailureDialog = true },
            enabled = canSaveFailureCase,
        ) {
            Text(
                if (canSaveFailureCase) "Save failure (${state.failureCasesSavedCount})"
                else "waiting for frame…",
            )
        }
        onTogglePause?.let {
            Button(onClick = it) {
                Text(if (isPaused) "Resume" else "Pause")
            }
        }
    }
}

/**
 * A compact button showing the active model name. Tapping it opens
 * [ModelPickerDialog] which lists all models with their AP and FPS
 * metadata so the user can make an informed choice without the chip
 * row consuming horizontal space on the live-camera screen.
 */
@Composable
private fun ModelSelectorButton(
    activeId: String,
    activeIsStubFallback: Boolean,
    onSelect: (String) -> Unit,
) {
    var showPicker by remember { mutableStateOf(false) }
    val activeSpec = ModelRegistry.byId(activeId)
    val label = when {
        activeIsStubFallback -> "⚠ ${activeSpec?.displayName ?: activeId}"
        else -> "Model: ${activeSpec?.displayName ?: activeId}"
    }
    Button(onClick = { showPicker = true }) {
        Text(text = label, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
    if (showPicker) {
        ModelPickerDialog(
            activeId = activeId,
            onSelect = { id ->
                showPicker = false
                onSelect(id)
            },
            onDismiss = { showPicker = false },
        )
    }
}

@Composable
private fun ModelPickerDialog(
    activeId: String,
    onSelect: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Select model") },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
            ) {
                // Header row
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Model", fontWeight = FontWeight.Bold, fontSize = 12.sp,
                        modifier = Modifier.weight(1f))
                    Text("AP", fontWeight = FontWeight.Bold, fontSize = 12.sp,
                        modifier = Modifier.width(40.dp))
                    Text("Speed", fontWeight = FontWeight.Bold, fontSize = 12.sp,
                        modifier = Modifier.width(80.dp))
                }
                HorizontalDivider()
                ModelRegistry.all.forEach { spec ->
                    ModelPickerRow(
                        spec = spec,
                        isActive = spec.modelId == activeId,
                        onClick = { onSelect(spec.modelId) },
                    )
                    HorizontalDivider(color = Color(0x33FFFFFF))
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}

@Composable
private fun ModelPickerRow(
    spec: ModelSpec,
    isActive: Boolean,
    onClick: () -> Unit,
) {
    val bg = if (isActive) Color(0x33FFFFFF) else Color.Transparent
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(bg)
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp, horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = if (isActive) "● ${spec.displayName}" else spec.displayName,
            fontSize = 13.sp,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f),
            fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
        )
        Text(
            text = spec.apAt50?.let { "${(it * 100).toInt()}%" } ?: "—",
            fontSize = 12.sp,
            modifier = Modifier.width(40.dp),
        )
        Text(
            text = spec.fpsHint ?: "—",
            fontSize = 12.sp,
            modifier = Modifier.width(80.dp),
        )
    }
}

@Composable
private fun ToggleRow(state: AppState) {
    Row(
        modifier = Modifier
            .background(Color(0x88000000))
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("HUD", color = Color.White)
        Switch(
            checked = state.showFps,
            onCheckedChange = { state.showFps = it },
        )
        Text("boxes", color = Color.White)
        Switch(
            checked = state.showOverlay,
            onCheckedChange = { state.showOverlay = it },
        )
        Text("front", color = Color.White)
        Switch(
            checked = state.useFrontCamera,
            onCheckedChange = { state.useFrontCamera = it },
        )
    }
}

@Composable
private fun ScoreThresholdControl(state: AppState) {
    Column(
        modifier = Modifier
            .background(Color(0x88000000))
            .padding(8.dp),
    ) {
        val rounded = (state.scoreThreshold * 100f).toInt() / 100f
        Text(
            "score ≥ ${(rounded * 100).toInt()}%",
            color = Color.White,
        )
        Slider(
            value = state.scoreThreshold,
            onValueChange = { state.scoreThreshold = it.coerceIn(0f, 1f) },
            valueRange = 0f..1f,
        )
    }
}

/**
 * Failure-type picker rendered as an AlertDialog. The previous
 * implementation stacked the picker above the bottom control bar in
 * a Column that grew past the screen on short phones; the dialog
 * floats over the camera preview and handles its own scrolling and
 * system-bar insets.
 */
@Composable
private fun FailureTypeDialog(
    onPick: (FailureType, String?) -> Unit,
    onDismiss: () -> Unit,
) {
    var note by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Save failure case") },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                TextField(
                    value = note,
                    onValueChange = { note = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("note (optional)") },
                    singleLine = false,
                )
                FailureType.values().forEach { ft ->
                    Button(
                        onClick = { onPick(ft, note.trim().ifEmpty { null }) },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(ft.name)
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) {
                Text("cancel")
            }
        },
    )
}
