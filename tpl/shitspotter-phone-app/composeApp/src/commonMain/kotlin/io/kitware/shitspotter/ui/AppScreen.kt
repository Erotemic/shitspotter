package io.kitware.shitspotter.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.runtime.LaunchedEffect
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
import io.kitware.shitspotter.core.CaptureReviewEntry
import io.kitware.shitspotter.core.MetadataMode
import io.kitware.shitspotter.core.ModelRegistry
import io.kitware.shitspotter.core.ModelSpec
import kotlinx.coroutines.delay

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
    onTakePhoto: ((CaptureLabel, String?) -> Unit)? = null,
    onToggleTorch: (() -> Unit)? = null,
    torchOn: Boolean = false,
    onReviewPhotos: (() -> Unit)? = null,
    onUpdatePhotoLabel: ((filePath: String, label: CaptureLabel, note: String?) -> Unit)? = null,
) {
    var flashTick by remember { mutableStateOf(0) }
    var flashVisible by remember { mutableStateOf(false) }
    LaunchedEffect(flashTick) {
        if (flashTick > 0) {
            flashVisible = true
            delay(120)
            flashVisible = false
        }
    }

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

            if (flashVisible) {
                Box(modifier = Modifier.fillMaxSize().background(Color.White.copy(alpha = 0.75f)))
            }

            // Top-left: HUD + score slider + errors
            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .systemBarsPadding()
                    .padding(8.dp)
                    .heightIn(max = 400.dp)
                    .verticalScroll(rememberScrollState()),
            ) {
                if (state.showFps) {
                    TelemetryHud(
                        telemetry = state.lastTelemetry,
                        detectionCount = state.lastDetections.size,
                    )
                    Spacer(Modifier.height(4.dp))
                }
                ScoreThresholdControl(state)
                state.lastError?.let { err ->
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "ERROR: $err",
                        color = Color(0xFFFF8080),
                        fontSize = 12.sp,
                        modifier = Modifier.background(Color(0x88000000)).padding(8.dp),
                    )
                }
            }

            // Top-right: compact model picker + settings gear
            val activeBackendName = state.lastTelemetry?.runtimeBackend
            val activeIsStubFallback =
                state.activeModelId != "stub-fake-detector" &&
                    activeBackendName != null &&
                    activeBackendName.startsWith("stub-")
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .systemBarsPadding()
                    .padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                ModelIconButton(
                    activeId = state.activeModelId,
                    activeIsStubFallback = activeIsStubFallback,
                    onSelect = { state.activeModelId = it },
                )
                SettingsIconButton(state)
            }

            if (activeIsStubFallback) {
                Text(
                    text = "⚠ model not on device — push .onnx to models/ and retap",
                    color = Color(0xFFFFCC66),
                    fontSize = 11.sp,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .systemBarsPadding()
                        .padding(top = 52.dp)
                        .background(Color(0xAA000000))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                )
            }

            // Bottom control bar
            CameraControlBar(
                onCapture = if (onTakePhoto != null) {
                    {
                        flashTick++
                        onTakePhoto(CaptureLabel.UNCERTAIN, null)
                    }
                } else null,
                onToggleTorch = onToggleTorch,
                torchOn = torchOn,
                photoCount = state.photosSavedCount,
                onReviewPhotos = onReviewPhotos,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .systemBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            )

            if (state.reviewMode) {
                ReviewScreen(
                    photos = state.capturedPhotos,
                    onClose = { state.reviewMode = false },
                    onUpdateLabel = onUpdatePhotoLabel,
                )
            }
        }
    }
}

@Composable
private fun CameraControlBar(
    onCapture: (() -> Unit)?,
    onToggleTorch: (() -> Unit)?,
    torchOn: Boolean,
    photoCount: Int,
    onReviewPhotos: (() -> Unit)?,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = Alignment.Center,
    ) {
        // Left: review button
        Box(modifier = Modifier.align(Alignment.CenterStart)) {
            if (onReviewPhotos != null) {
                IconButton(
                    onClick = onReviewPhotos,
                    modifier = Modifier
                        .size(56.dp)
                        .background(Color(0x55FFFFFF), CircleShape),
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("🖼", fontSize = 20.sp)
                        if (photoCount > 0) {
                            Text("$photoCount", fontSize = 9.sp, color = Color.White)
                        }
                    }
                }
            }
        }

        // Center: shutter FAB
        if (onCapture != null) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(80.dp)
                    .background(Color.White, CircleShape)
                    .clickable { onCapture() },
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    modifier = Modifier
                        .size(68.dp)
                        .background(Color(0xFFDDDDDD), CircleShape),
                )
            }
        } else {
            Spacer(Modifier.size(80.dp))
        }

        // Right: torch
        Box(modifier = Modifier.align(Alignment.CenterEnd)) {
            if (onToggleTorch != null) {
                IconButton(
                    onClick = onToggleTorch,
                    modifier = Modifier
                        .size(56.dp)
                        .background(
                            if (torchOn) Color(0xFFFFDD44) else Color(0x55FFFFFF),
                            CircleShape,
                        ),
                ) {
                    Text(
                        text = "🔦",
                        fontSize = 20.sp,
                        color = if (torchOn) Color.Black else Color(0xCCFFFFFF),
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelIconButton(
    activeId: String,
    activeIsStubFallback: Boolean,
    onSelect: (String) -> Unit,
) {
    var showPicker by remember { mutableStateOf(false) }
    IconButton(
        onClick = { showPicker = true },
        modifier = Modifier
            .size(40.dp)
            .background(
                if (activeIsStubFallback) Color(0x88FF8800) else Color(0x55FFFFFF),
                CircleShape,
            ),
    ) {
        Text(
            text = if (activeIsStubFallback) "⚠" else "🤖",
            fontSize = 16.sp,
        )
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
private fun SettingsIconButton(state: AppState) {
    var showSettings by remember { mutableStateOf(false) }
    IconButton(
        onClick = { showSettings = true },
        modifier = Modifier
            .size(40.dp)
            .background(Color(0x55FFFFFF), CircleShape),
    ) {
        Text("⚙", fontSize = 16.sp)
    }
    if (showSettings) {
        SettingsDialog(state = state, onDismiss = { showSettings = false })
    }
}

@Composable
private fun SettingsDialog(state: AppState, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                ToggleItem("Show telemetry HUD", state.showFps) { state.showFps = it }
                ToggleItem("Show detection boxes", state.showOverlay) { state.showOverlay = it }
                ToggleItem("Use front camera", state.useFrontCamera) { state.useFrontCamera = it }
                HorizontalDivider()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("Photo metadata", color = Color.White, fontSize = 14.sp)
                    MetadataModeSelector(state)
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Done") }
        },
    )
}

@Composable
private fun ToggleItem(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, color = Color.White, fontSize = 14.sp)
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun MetadataModeSelector(state: AppState) {
    val label = when (state.metadataMode) {
        MetadataMode.FULL -> "GPS on"
        MetadataMode.NO_GPS -> "GPS off"
        MetadataMode.NONE -> "No metadata"
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
            .padding(horizontal = 8.dp, vertical = 4.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Sensitive", color = Color(0xFFAABBFF), fontSize = 11.sp)
            Text(
                "${(state.scoreThreshold * 100).toInt()}%",
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
            )
            Text("Precise", color = Color(0xFFFFBBAA), fontSize = 11.sp)
        }
        Slider(
            value = state.scoreThreshold,
            onValueChange = { state.scoreThreshold = it.coerceIn(0f, 1f) },
            valueRange = 0f..1f,
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
                    Text("Model", fontWeight = FontWeight.Bold, fontSize = 12.sp, modifier = Modifier.weight(1f))
                    Text("AP", fontWeight = FontWeight.Bold, fontSize = 12.sp, modifier = Modifier.weight(0.22f))
                    Text("Speed", fontWeight = FontWeight.Bold, fontSize = 12.sp, modifier = Modifier.weight(0.44f))
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
private fun ModelPickerRow(spec: ModelSpec, isActive: Boolean, onClick: () -> Unit) {
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
            modifier = Modifier.weight(0.22f),
        )
        Text(
            text = spec.fpsHint ?: "—",
            fontSize = 12.sp,
            modifier = Modifier.weight(0.44f),
        )
    }
}

@Composable
private fun ReviewScreen(
    photos: List<CaptureReviewEntry>,
    onClose: () -> Unit,
    onUpdateLabel: ((String, CaptureLabel, String?) -> Unit)?,
) {
    Box(modifier = Modifier.fillMaxSize().background(Color(0xEE101010))) {
        Column(modifier = Modifier.fillMaxSize().systemBarsPadding()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Photos (${photos.size})",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onClose) { Text("Close", color = Color.White) }
            }
            HorizontalDivider()
            if (photos.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No photos captured yet.", color = Color(0xFF888888), fontSize = 14.sp)
                }
            } else {
                LazyColumn(modifier = Modifier.fillMaxSize()) {
                    items(photos, key = { it.filePath }) { entry ->
                        ReviewPhotoRow(entry = entry, onUpdateLabel = onUpdateLabel)
                        HorizontalDivider(color = Color(0x33FFFFFF))
                    }
                }
            }
        }
    }
}

@Composable
private fun ReviewPhotoRow(
    entry: CaptureReviewEntry,
    onUpdateLabel: ((String, CaptureLabel, String?) -> Unit)?,
) {
    var showPicker by remember { mutableStateOf(false) }
    if (showPicker) {
        LabelPickerDialog(
            current = entry.label,
            onPick = { label, note ->
                showPicker = false
                onUpdateLabel?.invoke(entry.filePath, label, note)
            },
            onDismiss = { showPicker = false },
        )
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(enabled = onUpdateLabel != null) { showPicker = true }
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                entry.timestamp.take(19).replace("T", " "),
                color = Color.White,
                fontSize = 13.sp,
            )
            Text(
                "${entry.detectionCount} detection${if (entry.detectionCount == 1) "" else "s"}",
                color = Color(0xFFAAAAAA),
                fontSize = 11.sp,
            )
            entry.note?.let {
                Text(it, color = Color(0xFFBBBBBB), fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
        Text(
            text = labelDisplayText(entry.label),
            color = labelDisplayColor(entry.label),
            fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier
                .background(Color(0x44FFFFFF))
                .padding(horizontal = 8.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun LabelPickerDialog(
    current: CaptureLabel,
    onPick: (CaptureLabel, String?) -> Unit,
    onDismiss: () -> Unit,
) {
    var note by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Annotate photo") },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(6.dp),
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
                        onClick = { onPick(label, note.trim().ifEmpty { null }) },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(labelDisplayText(label), color = labelDisplayColor(label))
                            if (label == current) Text("✓", color = Color.White)
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}

private fun labelDisplayText(label: CaptureLabel): String = when (label) {
    CaptureLabel.TRUE_POSITIVE -> "TP — correct detection"
    CaptureLabel.FALSE_POSITIVE -> "FP — spurious detection"
    CaptureLabel.FALSE_NEGATIVE -> "FN — missed detection"
    CaptureLabel.TRUE_NEGATIVE -> "TN — correctly empty"
    CaptureLabel.UNCERTAIN -> "? — uncertain"
}

private fun labelDisplayColor(label: CaptureLabel): Color = when (label) {
    CaptureLabel.TRUE_POSITIVE -> Color(0xFF66FF88)
    CaptureLabel.FALSE_POSITIVE -> Color(0xFFFF6666)
    CaptureLabel.FALSE_NEGATIVE -> Color(0xFFFFAA44)
    CaptureLabel.TRUE_NEGATIVE -> Color(0xFF88CCFF)
    CaptureLabel.UNCERTAIN -> Color(0xFFCCCCCC)
}
