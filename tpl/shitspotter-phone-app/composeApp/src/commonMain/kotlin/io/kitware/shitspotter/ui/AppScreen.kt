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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
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
import io.kitware.shitspotter.core.CaptureLabel
import io.kitware.shitspotter.core.FailureType
import io.kitware.shitspotter.core.MetadataMode
import io.kitware.shitspotter.core.ModelRegistry
import io.kitware.shitspotter.core.ModelSpec

interface CameraSurface {
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
    onTakePhoto: ((CaptureLabel, String?) -> Unit)? = null,
    onToggleTorch: (() -> Unit)? = null,
    torchOn: Boolean = false,
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

            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .systemBarsPadding()
                    .padding(8.dp)
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

            CameraControlBar(
                state = state,
                onSaveFailureCase = onSaveFailureCase,
                onTogglePause = onTogglePause,
                isPaused = isPaused,
                canSaveFailureCase = canSaveFailureCase,
                onTakePhoto = onTakePhoto,
                onToggleTorch = onToggleTorch,
                torchOn = torchOn,
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
private fun CameraControlBar(
    state: AppState,
    onSaveFailureCase: (FailureType, String?) -> Unit,
    onTogglePause: (() -> Unit)?,
    isPaused: Boolean,
    canSaveFailureCase: Boolean,
    onTakePhoto: ((CaptureLabel, String?) -> Unit)?,
    onToggleTorch: (() -> Unit)?,
    torchOn: Boolean,
    modifier: Modifier = Modifier,
) {
    var showFailureDialog by remember { mutableStateOf(false) }
    var showPhotoDialog by remember { mutableStateOf(false) }

    if (showFailureDialog) {
        FailureTypeDialog(
            onPick = { type, note ->
                showFailureDialog = false
                onSaveFailureCase(type, note)
            },
            onDismiss = { showFailureDialog = false },
        )
    }
    if (showPhotoDialog) {
        PhotoCaptureDialog(
            detectionCount = state.lastDetections.size,
            onCapture = { label, note ->
                showPhotoDialog = false
                onTakePhoto?.invoke(label, note)
            },
            onDismiss = { showPhotoDialog = false },
        )
    }

    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        // Left side: pause + legacy failure save
        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            onTogglePause?.let {
                Button(onClick = it) {
                    Text(if (isPaused) "Resume" else "Pause")
                }
            }
            Button(
                onClick = { showFailureDialog = true },
                enabled = canSaveFailureCase,
            ) {
                Text(
                    if (canSaveFailureCase) "Failure (${state.failureCasesSavedCount})"
                    else "…",
                )
            }
        }

        // Center: large photo capture FAB (only when onTakePhoto is wired)
        if (onTakePhoto != null) {
            Box(
                modifier = Modifier
                    .size(72.dp)
                    .background(Color(0xFFFFFFFF), CircleShape)
                    .clickable { showPhotoDialog = true },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = if (state.photosSavedCount > 0) "${state.photosSavedCount}" else "📷",
                    fontSize = 22.sp,
                    color = Color.Black,
                )
            }
        } else {
            Spacer(Modifier.width(72.dp))
        }

        // Right side: torch toggle (Android-only, shown only when lambda is non-null)
        if (onToggleTorch != null) {
            IconButton(
                onClick = onToggleTorch,
                modifier = Modifier
                    .size(48.dp)
                    .background(
                        if (torchOn) Color(0xFFFFDD44) else Color(0x55FFFFFF),
                        CircleShape,
                    ),
            ) {
                Text(
                    text = if (torchOn) "🔦" else "🔦",
                    fontSize = 20.sp,
                    color = if (torchOn) Color.Black else Color(0xCCFFFFFF),
                )
            }
        } else {
            Spacer(Modifier.width(48.dp))
        }
    }
}

@Composable
private fun PhotoCaptureDialog(
    detectionCount: Int,
    onCapture: (CaptureLabel, String?) -> Unit,
    onDismiss: () -> Unit,
) {
    var note by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Save photo (${detectionCount} detection${if (detectionCount == 1) "" else "s"})") },
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
                CaptureLabel.values().forEach { label ->
                    Button(
                        onClick = { onCapture(label, note.trim().ifEmpty { null }) },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(label.name)
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) { Text("cancel") }
        },
    )
}

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
        Spacer(Modifier.width(4.dp))
        MetadataModeSelector(state)
    }
}

@Composable
private fun MetadataModeSelector(state: AppState) {
    val label = when (state.metadataMode) {
        MetadataMode.FULL -> "GPS:on"
        MetadataMode.NO_GPS -> "GPS:off"
        MetadataMode.NONE -> "meta:none"
    }
    TextButton(
        onClick = {
            state.metadataMode = when (state.metadataMode) {
                MetadataMode.FULL -> MetadataMode.NO_GPS
                MetadataMode.NO_GPS -> MetadataMode.NONE
                MetadataMode.NONE -> MetadataMode.FULL
            }
        },
        modifier = Modifier.background(Color(0x44FFFFFF)),
    ) {
        Text(label, color = Color.White, fontSize = 12.sp)
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
