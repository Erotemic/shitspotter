package io.github.erotemic.shitspotter.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
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
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.erotemic.shitspotter.core.AppLogger
import io.github.erotemic.shitspotter.core.AppState
import io.github.erotemic.shitspotter.core.BoundingBox
import io.github.erotemic.shitspotter.core.PrintlnLogger
import io.github.erotemic.shitspotter.core.CaptureLabel
import io.github.erotemic.shitspotter.core.CaptureMetadata
import io.github.erotemic.shitspotter.core.CaptureReviewEntry
import io.github.erotemic.shitspotter.core.Detection
import io.github.erotemic.shitspotter.core.DetectionAnnotation
import io.github.erotemic.shitspotter.core.MetadataMode
import io.github.erotemic.shitspotter.core.ModelFormat
import io.github.erotemic.shitspotter.core.ModelRegistry
import io.github.erotemic.shitspotter.core.ModelSpec
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

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
    onSharePhoto: ((filePath: String) -> Unit)? = null,
    onShareAllPhotos: (() -> Unit)? = null,
    onSharePhotos: ((filePaths: List<String>) -> Unit)? = null,
    onDeletePhoto: ((filePath: String) -> Unit)? = null,
    onLoadMetadata: (suspend (String) -> CaptureMetadata?)? = null,
    onUpdateDetectionAnnotations: ((String, Map<String, DetectionAnnotation>, List<BoundingBox>) -> Unit)? = null,
    logger: AppLogger = PrintlnLogger,
) {
    // System back: close photo viewer first, then review screen; else let system handle.
    PlatformBackHandler(enabled = state.reviewMode && state.viewingPhotoPath != null) {
        state.viewingPhotoPath = null
    }
    PlatformBackHandler(enabled = state.reviewMode && state.viewingPhotoPath == null) {
        state.reviewMode = false
    }

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

            DetectionOverlay(
                detections = state.lastDetections,
                frameWidth = state.lastFrameWidth,
                frameHeight = state.lastFrameHeight,
                modifier = Modifier.fillMaxSize(),
                scaleMode = cameraSurface.overlayScaleMode,
            )

            if (flashVisible) {
                Box(modifier = Modifier.fillMaxSize().background(Color.White.copy(alpha = 0.75f)))
            }

            // Top area: HUD + slider on the left, single ⚙ button top-right
            val activeBackendName = state.lastTelemetry?.runtimeBackend
            val activeSpec = ModelRegistry.byId(state.activeModelId)
            val activeIsStubFallback =
                activeSpec != null &&
                    activeSpec.format != ModelFormat.STUB &&
                    activeBackendName != null &&
                    activeBackendName.startsWith("stub-")
            Row(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .fillMaxWidth()
                    .systemBarsPadding()
                    .padding(start = 8.dp, end = 8.dp, top = 8.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(max = 400.dp)
                        .verticalScroll(rememberScrollState())
                        .padding(end = 16.dp),
                ) {
                    if (state.useOfficialName) {
                        Text(
                            "ShitSpotter",
                            color = Color(0xFFFFCC00),
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                        )
                        Spacer(Modifier.height(2.dp))
                    }
                    if (state.showFps) {
                        TelemetryHud(
                            telemetry = state.lastTelemetry,
                            detectionCount = state.lastDetections.size,
                        )
                        Spacer(Modifier.height(4.dp))
                    }
                    if (state.showScoreSlider) {
                        ScoreThresholdControl(state)
                    }
                    if (activeIsStubFallback) {
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = "⚠ model not on device — push .onnx to models/ and retap",
                            color = Color(0xFFFFCC66),
                            fontSize = 11.sp,
                            modifier = Modifier
                                .background(Color(0xAA000000))
                                .padding(horizontal = 8.dp, vertical = 4.dp),
                        )
                    }
                    state.lastError?.let { err ->
                        Spacer(Modifier.height(4.dp))
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

                // Single settings icon — top-right
                SettingsIconButton(state, activeIsStubFallback)
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
                    viewingPhotoPath = state.viewingPhotoPath,
                    onClose = { state.reviewMode = false },
                    onViewPhoto = { state.viewingPhotoPath = it },
                    onCloseViewer = { state.viewingPhotoPath = null },
                    onUpdateLabel = onUpdatePhotoLabel,
                    onSharePhoto = onSharePhoto,
                    onShareAllPhotos = onShareAllPhotos,
                    onSharePhotos = onSharePhotos,
                    onDeletePhoto = onDeletePhoto,
                    onLoadMetadata = onLoadMetadata,
                    onUpdateDetectionAnnotations = onUpdateDetectionAnnotations,
                    logger = logger,
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
        Box(modifier = Modifier.align(Alignment.CenterStart)) {
            if (onReviewPhotos != null) {
                SmallCircleButton(
                    onClick = onReviewPhotos,
                    circleSize = 32.dp,
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("🖼", fontSize = 20.sp)
                        if (photoCount > 0) {
                            Text("$photoCount", fontSize = 8.sp, color = Color.White, lineHeight = 8.sp)
                        }
                    }
                }
            }
        }

        if (onCapture != null) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(60.dp)
                    .background(Color.White, CircleShape)
                    .clickable { onCapture() },
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    modifier = Modifier
                        .size(51.dp)
                        .background(Color(0xFFDDDDDD), CircleShape),
                )
            }
        } else {
            Spacer(Modifier.size(60.dp))
        }

        Box(modifier = Modifier.align(Alignment.CenterEnd)) {
            if (onToggleTorch != null) {
                SmallCircleButton(
                    onClick = onToggleTorch,
                    circleColor = if (torchOn) Color(0xFFFFDD44) else Color(0x44FFFFFF),
                    circleSize = 32.dp,
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

/**
 * Icon button where the circle background is SMALLER than the tap target so
 * the emoji glyph is never clipped by its container. The outer [tapSize] box
 * is the actual touch area; [circleSize] is only the painted background disc.
 */
@Composable
private fun SmallCircleButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    circleColor: Color = Color(0x44FFFFFF),
    circleSize: androidx.compose.ui.unit.Dp = 32.dp,
    tapSize: androidx.compose.ui.unit.Dp = 40.dp,
    content: @Composable () -> Unit,
) {
    Box(
        modifier = modifier
            .size(tapSize)
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = onClick,
            ),
        contentAlignment = Alignment.Center,
    ) {
        Box(modifier = Modifier.size(circleSize).background(circleColor, CircleShape))
        content()
    }
}

@Composable
private fun SettingsIconButton(state: AppState, activeIsStubFallback: Boolean) {
    var showSettings by remember { mutableStateOf(false) }
    SmallCircleButton(
        onClick = { showSettings = true },
        circleColor = if (activeIsStubFallback) Color(0x88FF8800) else Color(0x44FFFFFF),
        circleSize = 32.dp,
        tapSize = 40.dp,
    ) {
        Text(if (activeIsStubFallback) "⚠" else "⚙", fontSize = 20.sp)
    }
    if (showSettings) {
        SettingsDialog(state = state, onDismiss = { showSettings = false })
    }
}

@Composable
private fun SettingsDialog(state: AppState, onDismiss: () -> Unit) {
    var showModelPicker by remember { mutableStateOf(false) }

    if (showModelPicker) {
        ModelPickerDialog(
            activeId = state.activeModelId,
            onSelect = { id ->
                state.activeModelId = id
                showModelPicker = false
            },
            onDismiss = { showModelPicker = false },
        )
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                // Model selector row
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { showModelPicker = true }
                        .background(Color(0x22FFFFFF))
                        .padding(horizontal = 8.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("Model", color = Color.White, fontSize = 14.sp)
                    val spec = ModelRegistry.byId(state.activeModelId)
                    Text(
                        spec?.displayName ?: state.activeModelId,
                        color = Color(0xFFAACCFF),
                        fontSize = 12.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f, fill = false).padding(start = 8.dp),
                    )
                    Text(" ›", color = Color(0xFF888888), fontSize = 14.sp)
                }
                HorizontalDivider()
                ToggleItem("Show HUD", state.showFps) { state.showFps = it }
                ToggleItem("Show score slider", state.showScoreSlider) { state.showScoreSlider = it }
                ToggleItem("Use front camera", state.useFrontCamera) { state.useFrontCamera = it }
                ToggleItem("ShitSpotter mode", state.useOfficialName) { state.useOfficialName = it }
                HorizontalDivider()
                MetadataToggles(state)
                HorizontalDivider()
                TextField(
                    value = state.recipientEmail,
                    onValueChange = { state.recipientEmail = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Recipient email") },
                    placeholder = { Text("you@example.com") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                    supportingText = {
                        Text(
                            "Photos are only sent when you tap ✉ — capturing never sends automatically.",
                            fontSize = 11.sp,
                            color = Color(0xFF888888),
                        )
                    },
                )
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
private fun MetadataToggles(state: AppState) {
    val exifEnabled = state.metadataMode != MetadataMode.NONE
    val gpsEnabled = state.metadataMode == MetadataMode.FULL

    ToggleItem("Save EXIF metadata", exifEnabled) { on ->
        state.metadataMode = if (on) MetadataMode.NO_GPS else MetadataMode.NONE
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            "Include GPS location",
            color = if (exifEnabled) Color.White else Color(0xFF666666),
            fontSize = 14.sp,
        )
        Switch(
            checked = gpsEnabled,
            onCheckedChange = { on ->
                state.metadataMode = if (on) MetadataMode.FULL else MetadataMode.NO_GPS
            },
            enabled = exifEnabled,
        )
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
            modifier = Modifier.height(24.dp),
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

// ── Review screen ────────────────────────────────────────────────────────────

private enum class ReviewSortOrder(val label: String) {
    DATE_DESC("Date ↓"),
    DATE_ASC("Date ↑"),
    DETECTIONS_DESC("Detections ↓"),
}

@Composable
private fun ReviewScreen(
    photos: List<CaptureReviewEntry>,
    viewingPhotoPath: String?,
    onClose: () -> Unit,
    onViewPhoto: (String) -> Unit,
    onCloseViewer: () -> Unit,
    onUpdateLabel: ((String, CaptureLabel, String?) -> Unit)?,
    onSharePhoto: ((String) -> Unit)?,
    onShareAllPhotos: (() -> Unit)?,
    onSharePhotos: ((List<String>) -> Unit)?,
    onDeletePhoto: ((String) -> Unit)?,
    onLoadMetadata: (suspend (String) -> CaptureMetadata?)? = null,
    onUpdateDetectionAnnotations: ((String, Map<String, DetectionAnnotation>, List<BoundingBox>) -> Unit)? = null,
    logger: AppLogger = PrintlnLogger,
) {
    var selectionMode by remember { mutableStateOf(false) }
    var selectedPaths by remember { mutableStateOf(emptySet<String>()) }
    var pendingBulkLabel by remember { mutableStateOf<CaptureLabel?>(null) }
    var showBulkLabelPicker by remember { mutableStateOf(false) }
    var showBulkLabelConfirm by remember { mutableStateOf(false) }
    var showBulkDeleteConfirm by remember { mutableStateOf(false) }
    var sortOrder by remember { mutableStateOf(ReviewSortOrder.DATE_DESC) }
    val sortedPhotos = remember(photos, sortOrder) {
        when (sortOrder) {
            ReviewSortOrder.DATE_DESC -> photos.sortedByDescending { it.timestamp }
            ReviewSortOrder.DATE_ASC -> photos.sortedBy { it.timestamp }
            ReviewSortOrder.DETECTIONS_DESC -> photos.sortedByDescending { it.detectionCount }
        }
    }

    // Back exits selection mode before closing the review screen
    PlatformBackHandler(enabled = selectionMode) {
        selectionMode = false
        selectedPaths = emptySet()
    }

    if (showBulkLabelPicker) {
        LabelPickerDialog(
            current = CaptureLabel.UNCERTAIN,
            onPick = { label, _ ->
                pendingBulkLabel = label
                showBulkLabelPicker = false
                showBulkLabelConfirm = true
            },
            onDismiss = { showBulkLabelPicker = false },
        )
    }
    if (showBulkLabelConfirm && pendingBulkLabel != null) {
        val label = pendingBulkLabel!!
        AlertDialog(
            onDismissRequest = { showBulkLabelConfirm = false },
            title = { Text("Change ${selectedPaths.size} photo${if (selectedPaths.size == 1) "" else "s"}?") },
            text = {
                Text(
                    "Set label to \"${labelDisplayText(label)}\" on ${selectedPaths.size} selected photo${if (selectedPaths.size == 1) "" else "s"}.",
                    color = Color(0xFFCCCCCC),
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showBulkLabelConfirm = false
                    selectedPaths.forEach { path -> onUpdateLabel?.invoke(path, label, null) }
                    selectionMode = false
                    selectedPaths = emptySet()
                }) { Text("Apply", color = labelDisplayColor(label)) }
            },
            dismissButton = {
                TextButton(onClick = { showBulkLabelConfirm = false }) { Text("Cancel") }
            },
        )
    }
    if (showBulkDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showBulkDeleteConfirm = false },
            title = { Text("Delete ${selectedPaths.size} photo${if (selectedPaths.size == 1) "" else "s"}?") },
            text = { Text("This cannot be undone.", color = Color(0xFFCCCCCC)) },
            confirmButton = {
                TextButton(onClick = {
                    showBulkDeleteConfirm = false
                    selectedPaths.toList().forEach { path -> onDeletePhoto?.invoke(path) }
                    selectionMode = false
                    selectedPaths = emptySet()
                }) { Text("Delete", color = Color(0xFFFF6666)) }
            },
            dismissButton = {
                TextButton(onClick = { showBulkDeleteConfirm = false }) { Text("Cancel") }
            },
        )
    }

    Box(modifier = Modifier.fillMaxSize().background(Color(0xEE101010))) {
        Column(modifier = Modifier.fillMaxSize().systemBarsPadding()) {
            if (selectionMode) {
                // Selection mode header
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextButton(onClick = { selectionMode = false; selectedPaths = emptySet() }) {
                        Text("Cancel", color = Color(0xFF88CCFF))
                    }
                    Text(
                        "${selectedPaths.size} selected",
                        color = Color.White,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.Center,
                    )
                    // Select all / none toggle
                    val allSelected = sortedPhotos.isNotEmpty() && selectedPaths.size == sortedPhotos.size
                    TextButton(onClick = {
                        selectedPaths = if (allSelected) emptySet() else sortedPhotos.map { it.filePath }.toSet()
                    }) {
                        Text(if (allSelected) "None" else "All", color = Color(0xFF88CCFF))
                    }
                    if (selectedPaths.isNotEmpty()) {
                        if (onSharePhotos != null) {
                            IconButton(onClick = { onSharePhotos(selectedPaths.toList()) }) {
                                Text("✉", fontSize = 18.sp, color = Color(0xFF88CCFF))
                            }
                        }
                        if (onUpdateLabel != null) {
                            IconButton(onClick = { showBulkLabelPicker = true }) {
                                Text("✏", fontSize = 18.sp, color = Color.White)
                            }
                        }
                        if (onDeletePhoto != null) {
                            IconButton(onClick = { showBulkDeleteConfirm = true }) {
                                Text("🗑", fontSize = 18.sp, color = Color(0xFFFF6666))
                            }
                        }
                    }
                }
            } else {
                // Normal header
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "Photos (${photos.size})",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        modifier = Modifier.weight(1f),
                    )
                    if (photos.isNotEmpty()) {
                        TextButton(onClick = {
                            sortOrder = ReviewSortOrder.entries[(sortOrder.ordinal + 1) % ReviewSortOrder.entries.size]
                        }) { Text(sortOrder.label, color = Color(0xFF888888), fontSize = 12.sp) }
                        TextButton(onClick = { selectionMode = true }) {
                            Text("Select", color = Color(0xFF88CCFF))
                        }
                    }
                    if (onShareAllPhotos != null && photos.isNotEmpty()) {
                        TextButton(onClick = onShareAllPhotos) {
                            Text("✉ All", color = Color(0xFF88CCFF))
                        }
                    }
                    TextButton(onClick = onClose) { Text("Close", color = Color.White) }
                }
            }
            HorizontalDivider()
            if (sortedPhotos.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No photos captured yet.", color = Color(0xFF888888), fontSize = 14.sp)
                }
            } else {
                LazyColumn(modifier = Modifier.fillMaxSize()) {
                    items(sortedPhotos, key = { it.filePath }) { entry ->
                        val isSelected = entry.filePath in selectedPaths
                        ReviewPhotoRow(
                            entry = entry,
                            selectionMode = selectionMode,
                            selected = isSelected,
                            onTap = { onViewPhoto(entry.filePath) },
                            onToggleSelect = {
                                selectedPaths = if (isSelected) selectedPaths - entry.filePath
                                else selectedPaths + entry.filePath
                            },
                            onLongPress = {
                                selectionMode = true
                                selectedPaths = setOf(entry.filePath)
                            },
                            onUpdateLabel = onUpdateLabel,
                            onDelete = if (onDeletePhoto != null) {
                                { onDeletePhoto(entry.filePath) }
                            } else null,
                        )
                        HorizontalDivider(color = Color(0x33FFFFFF))
                    }
                }
            }
        }

        // Full-screen photo viewer layered on top of the list
        if (viewingPhotoPath != null && sortedPhotos.isNotEmpty()) {
            val initialIndex = sortedPhotos.indexOfFirst { it.filePath == viewingPhotoPath }.coerceAtLeast(0)
            PhotoViewer(
                photos = sortedPhotos,
                initialIndex = initialIndex,
                onClose = onCloseViewer,
                onUpdateLabel = { filePath, label, note ->
                    onUpdateLabel?.invoke(filePath, label, note)
                },
                onShare = onSharePhoto,
                onDelete = if (onDeletePhoto != null) {
                    { filePath -> onDeletePhoto(filePath); onCloseViewer() }
                } else null,
                onLoadMetadata = onLoadMetadata,
                onUpdateDetectionAnnotations = onUpdateDetectionAnnotations,
                logger = logger,
            )
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ReviewPhotoRow(
    entry: CaptureReviewEntry,
    selectionMode: Boolean,
    selected: Boolean,
    onTap: () -> Unit,
    onToggleSelect: () -> Unit,
    onLongPress: (() -> Unit)? = null,
    onUpdateLabel: ((String, CaptureLabel, String?) -> Unit)?,
    onDelete: (() -> Unit)? = null,
) {
    var showAnnotatePicker by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    if (showAnnotatePicker) {
        LabelPickerDialog(
            current = entry.label,
            onPick = { label, note ->
                showAnnotatePicker = false
                onUpdateLabel?.invoke(entry.filePath, label, note)
            },
            onDismiss = { showAnnotatePicker = false },
        )
    }
    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete photo?") },
            text = { Text("This cannot be undone.", color = Color(0xFFCCCCCC)) },
            confirmButton = {
                TextButton(onClick = { showDeleteConfirm = false; onDelete?.invoke() }) {
                    Text("Delete", color = Color(0xFFFF6666))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) { Text("Cancel") }
            },
        )
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(if (selected) Color(0x22AADDFF) else Color.Transparent)
            .combinedClickable(
                onClick = if (selectionMode) onToggleSelect else onTap,
                onLongClick = if (!selectionMode && onLongPress != null) onLongPress else null,
            )
            .padding(start = 12.dp, end = 4.dp, top = 10.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (selectionMode) {
            Box(
                modifier = Modifier
                    .size(22.dp)
                    .background(
                        if (selected) Color(0xFF88CCFF) else Color.Transparent,
                        CircleShape,
                    )
                    .then(
                        if (!selected) Modifier.background(Color(0x33FFFFFF), CircleShape) else Modifier,
                    ),
                contentAlignment = Alignment.Center,
            ) {
                if (selected) Text("✓", color = Color.Black, fontSize = 13.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.size(10.dp))
        }
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
                Text(
                    it,
                    color = Color(0xFFBBBBBB),
                    fontSize = 11.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
        // Label chip + delete button (outside selection mode)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .clickable(enabled = !selectionMode && onUpdateLabel != null) { showAnnotatePicker = true }
                    .background(Color(0x44FFFFFF))
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            ) {
                Text(
                    text = labelDisplayText(entry.label),
                    color = labelDisplayColor(entry.label),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            if (!selectionMode && onDelete != null) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                        ) { showDeleteConfirm = true },
                    contentAlignment = Alignment.Center,
                ) {
                    Text("🗑", fontSize = 16.sp, color = Color(0xFFFF6666))
                }
            }
        }
    }
}

@Composable
private fun PhotoViewer(
    photos: List<CaptureReviewEntry>,
    initialIndex: Int,
    onClose: () -> Unit,
    onUpdateLabel: (filePath: String, CaptureLabel, String?) -> Unit,
    onShare: ((filePath: String) -> Unit)? = null,
    onDelete: ((filePath: String) -> Unit)? = null,
    onLoadMetadata: (suspend (String) -> CaptureMetadata?)? = null,
    onUpdateDetectionAnnotations: ((String, Map<String, DetectionAnnotation>, List<BoundingBox>) -> Unit)? = null,
    logger: AppLogger = PrintlnLogger,
) {
    val pagerState = rememberPagerState(initialPage = initialIndex) { photos.size }
    var showAnnotatePicker by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }
    var drawingFnMode by remember { mutableStateOf(false) }
    var metadata by remember { mutableStateOf<CaptureMetadata?>(null) }
    var localAnnotations by remember { mutableStateOf<Map<String, DetectionAnnotation>>(emptyMap()) }
    var localMissedBoxes by remember { mutableStateOf<List<BoundingBox>>(emptyList()) }

    val currentEntry = photos.getOrNull(pagerState.currentPage) ?: return

    LaunchedEffect(currentEntry.filePath) {
        metadata = null
        localAnnotations = emptyMap()
        localMissedBoxes = emptyList()
        drawingFnMode = false
        val m = onLoadMetadata?.invoke(currentEntry.filePath)
        metadata = m
        localAnnotations = m?.detectionAnnotations ?: emptyMap()
        localMissedBoxes = m?.missedBoxes ?: emptyList()
    }

    if (showAnnotatePicker) {
        LabelPickerDialog(
            current = currentEntry.label,
            onPick = { label, note ->
                showAnnotatePicker = false
                onUpdateLabel(currentEntry.filePath, label, note)
            },
            onDismiss = { showAnnotatePicker = false },
        )
    }
    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete photo?") },
            text = { Text("This will permanently delete the photo and its metadata.", color = Color(0xFFCCCCCC)) },
            confirmButton = {
                TextButton(onClick = { showDeleteConfirm = false; onDelete?.invoke(currentEntry.filePath) }) {
                    Text("Delete", color = Color(0xFFFF6666))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) { Text("Cancel") }
            },
        )
    }

    val m = metadata

    // Zoom + pan state — reset whenever the current page changes
    var zoomScale by remember { mutableStateOf(1f) }
    var panOffset by remember { mutableStateOf(Offset.Zero) }
    LaunchedEffect(pagerState.currentPage) { zoomScale = 1f; panOffset = Offset.Zero }

    // Swipe target set from within the restricted awaitEachGesture scope (no suspend allowed
    // there); the LaunchedEffect runs outside and calls the actual suspend animateScrollToPage.
    var pendingSwipePage by remember { mutableStateOf<Int?>(null) }
    LaunchedEffect(pendingSwipePage) {
        val target = pendingSwipePage ?: return@LaunchedEffect
        if (target != pagerState.currentPage) pagerState.animateScrollToPage(target)
        pendingSwipePage = null
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            // PointerEventPass.Initial fires on the parent BEFORE any children see the event.
            // This guarantees we intercept 2-finger pinch and zoomed 1-finger pan before
            // HorizontalPager's internal scrollable processes them, while leaving single-
            // finger swipes at zoom=1 unconsumed so the pager handles page changes naturally.
            .pointerInput(Unit) {
                logger.info("Gesture", "handler installed")
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false, pass = PointerEventPass.Initial)
                    logger.info("Gesture", "gesture started zoom=$zoomScale")
                    var totalDx = 0f
                    var totalDy = 0f
                    var had2Fingers = false
                    while (true) {
                        val event = awaitPointerEvent(PointerEventPass.Initial)
                        val pressed = event.changes.filter { it.pressed }
                        if (pressed.isEmpty()) break
                        if (pressed.size >= 2) {
                            had2Fingers = true
                            logger.info("Gesture", "pinch ${pressed.size}pt zoom=$zoomScale")
                            val a = pressed[0]; val b = pressed[1]
                            val curr = (a.position - b.position).getDistance()
                            val prev = (a.previousPosition - b.previousPosition).getDistance()
                            if (prev > 0f && curr > 0f) {
                                zoomScale = (zoomScale * curr / prev).coerceIn(1f, 8f)
                            }
                            val centroidDelta = (a.position + b.position) / 2f -
                                (a.previousPosition + b.previousPosition) / 2f
                            if (zoomScale > 1f) panOffset += centroidDelta
                            event.changes.forEach { it.consume() }
                        } else {
                            val p = pressed.first()
                            val delta = p.position - p.previousPosition
                            if (zoomScale > 1.05f) {
                                panOffset += delta
                                event.changes.forEach { it.consume() }
                            } else {
                                totalDx += delta.x
                                totalDy += delta.y
                            }
                        }
                    }
                    // Gesture ended — detect horizontal swipe and manually page
                    val swipeThresh = viewConfiguration.touchSlop * 8f
                    logger.info("Gesture", "ended zoom=$zoomScale dx=$totalDx dy=$totalDy 2f=$had2Fingers")
                    if (!had2Fingers && zoomScale <= 1.05f &&
                        abs(totalDx) > swipeThresh && abs(totalDx) > abs(totalDy) * 1.5f
                    ) {
                        val target = if (totalDx < 0) {
                            (pagerState.currentPage + 1).coerceAtMost(photos.size - 1)
                        } else {
                            (pagerState.currentPage - 1).coerceAtLeast(0)
                        }
                        logger.info("Gesture", "swipe → page $target (was ${pagerState.currentPage})")
                        pendingSwipePage = target
                    }
                }
            },
    ) {
        HorizontalPager(
            state = pagerState,
            modifier = Modifier.fillMaxSize(),
            userScrollEnabled = false,
        ) { page ->
            val entry = photos.getOrNull(page) ?: return@HorizontalPager
            var bitmap by remember(entry.filePath) { mutableStateOf<ImageBitmap?>(null) }
            LaunchedEffect(entry.filePath) {
                bitmap = withContext(Dispatchers.IO) { loadImageBitmapFromFile(entry.filePath) }
            }
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer {
                        scaleX = zoomScale
                        scaleY = zoomScale
                        translationX = panOffset.x
                        translationY = panOffset.y
                    },
                contentAlignment = Alignment.Center,
            ) {
                val bmp = bitmap
                if (bmp != null) {
                    Image(
                        bitmap = bmp,
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit,
                    )
                } else {
                    CircularProgressIndicator(color = Color.White)
                }
            }
        }

        // Detection annotation overlay — transforms with the image via the same graphicsLayer.
        // zoomScale/panOffset are passed through so the tap handler can inverse-transform
        // touch coordinates back to pre-zoom layout space for correct box hit testing.
        if (m != null && m.frameWidth != null && m.frameHeight != null) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer {
                        scaleX = zoomScale
                        scaleY = zoomScale
                        translationX = panOffset.x
                        translationY = panOffset.y
                    },
            ) {
                PhotoDetectionOverlay(
                    detections = m.detections,
                    frameWidth = m.frameWidth,
                    frameHeight = m.frameHeight,
                    annotations = localAnnotations,
                    missedBoxes = localMissedBoxes,
                    drawingFnMode = drawingFnMode,
                    zoomScale = zoomScale,
                    panOffset = panOffset,
                    onAnnotationChanged = { key, ann ->
                        val updated = if (ann == null) localAnnotations - key else localAnnotations + (key to ann)
                        localAnnotations = updated
                        onUpdateDetectionAnnotations?.invoke(currentEntry.filePath, updated, localMissedBoxes)
                    },
                    onMissedBoxAdded = { box ->
                        val updated = localMissedBoxes + box
                        localMissedBoxes = updated
                        drawingFnMode = false
                        onUpdateDetectionAnnotations?.invoke(currentEntry.filePath, localAnnotations, updated)
                    },
                    onMissedBoxRemoved = { idx ->
                        val updated = localMissedBoxes.toMutableList().also { it.removeAt(idx) }
                        localMissedBoxes = updated
                        onUpdateDetectionAnnotations?.invoke(currentEntry.filePath, localAnnotations, updated)
                    },
                )
            }
        }

        // Page indicator dots (when > 1 photo)
        if (photos.size > 1) {
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .systemBarsPadding()
                    .padding(bottom = 96.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                val displayCount = minOf(photos.size, 9)
                repeat(displayCount) { i ->
                    val active = i == pagerState.currentPage.coerceIn(0, displayCount - 1)
                    Box(
                        modifier = Modifier
                            .size(if (active) 8.dp else 5.dp)
                            .background(if (active) Color.White else Color(0x88FFFFFF), CircleShape),
                    )
                }
                if (photos.size > 9) {
                    Text("…", color = Color(0x88FFFFFF), fontSize = 12.sp)
                }
            }
        }

        // Top bar: close (left) + share + delete (right)
        Row(
            modifier = Modifier
                .align(Alignment.TopStart)
                .fillMaxWidth()
                .systemBarsPadding()
                .padding(8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            SmallCircleButton(onClick = onClose, circleColor = Color(0x88000000), circleSize = 32.dp) {
                Text("✕", color = Color.White, fontSize = 20.sp)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                if (onShare != null) {
                    SmallCircleButton(
                        onClick = { onShare(currentEntry.filePath) },
                        circleColor = Color(0x88000000),
                        circleSize = 32.dp,
                    ) { Text("✉", color = Color(0xFF88CCFF), fontSize = 20.sp) }
                }
                if (onDelete != null) {
                    SmallCircleButton(
                        onClick = { showDeleteConfirm = true },
                        circleColor = Color(0x88000000),
                        circleSize = 32.dp,
                    ) { Text("🗑", color = Color(0xFFFF6666), fontSize = 20.sp) }
                }
            }
        }

        // Bottom: photo label + annotate + draw-FN toggle
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .systemBarsPadding()
                .padding(bottom = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                text = labelDisplayText(currentEntry.label),
                color = labelDisplayColor(currentEntry.label),
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.background(Color(0xAA000000)).padding(horizontal = 12.dp, vertical = 6.dp),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { showAnnotatePicker = true }) { Text("Annotate") }
                if (m?.frameWidth != null) {
                    Button(
                        onClick = { drawingFnMode = !drawingFnMode },
                        colors = androidx.compose.material3.ButtonDefaults.buttonColors(
                            containerColor = if (drawingFnMode) Color(0xFF7755BB) else Color(0xFF335566),
                        ),
                    ) { Text(if (drawingFnMode) "Cancel draw" else "+ Mark miss") }
                }
            }
            when {
                drawingFnMode -> Text(
                    "Drag to draw a missed-detection box",
                    color = Color(0xFFCCCCFF), fontSize = 11.sp,
                    modifier = Modifier.background(Color(0xAA000000)).padding(horizontal = 8.dp, vertical = 3.dp),
                )
                m?.frameWidth != null && m.detections.isNotEmpty() -> Text(
                    "Tap box: yellow=? → green=TP → red=FP",
                    color = Color(0xFFAAAAAA), fontSize = 10.sp,
                    modifier = Modifier.background(Color(0xAA000000)).padding(horizontal = 8.dp, vertical = 3.dp),
                )
            }
        }
    }
}

@Composable
private fun PhotoDetectionOverlay(
    detections: List<Detection>,
    frameWidth: Int,
    frameHeight: Int,
    annotations: Map<String, DetectionAnnotation>,
    missedBoxes: List<BoundingBox>,
    drawingFnMode: Boolean,
    onAnnotationChanged: (String, DetectionAnnotation?) -> Unit,
    onMissedBoxAdded: (BoundingBox) -> Unit,
    onMissedBoxRemoved: (Int) -> Unit,
    zoomScale: Float = 1f,
    panOffset: Offset = Offset.Zero,
) {
    val textMeasurer = rememberTextMeasurer()
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }
    var dragStart by remember { mutableStateOf<Offset?>(null) }
    var dragCurrent by remember { mutableStateOf<Offset?>(null) }

    Box(modifier = Modifier.fillMaxSize().onSizeChanged { canvasSize = it }) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            if (canvasSize.width <= 0 || frameWidth <= 0 || frameHeight <= 0) return@Canvas
            val sw = canvasSize.width.toFloat()
            val sh = canvasSize.height.toFloat()
            val scale = min(sw / frameWidth, sh / frameHeight)
            val ox = (sw - frameWidth * scale) / 2f
            val oy = (sh - frameHeight * scale) / 2f

            // Existing model detections
            detections.forEachIndexed { i, det ->
                val ann = annotations["$i"]
                val color = when (ann) {
                    DetectionAnnotation.TRUE_POSITIVE -> Color(0xFF44CC77)
                    DetectionAnnotation.FALSE_POSITIVE -> Color(0xFFCC4444)
                    null -> Color(0xFFFFDD22)
                }
                val l = ox + det.box.left * scale
                val t = oy + det.box.top * scale
                drawRect(color, topLeft = Offset(l, t), size = Size(det.box.width * scale, det.box.height * scale), style = Stroke(4f))
                val labelText = when (ann) {
                    DetectionAnnotation.TRUE_POSITIVE -> "TP"
                    DetectionAnnotation.FALSE_POSITIVE -> "FP"
                    null -> "${(det.score * 100).toInt()}%"
                }
                val measured = textMeasurer.measure(labelText, TextStyle(color = Color.White, fontSize = 11.sp))
                val lx = l.coerceAtMost(sw - measured.size.width - 4f)
                val ly = (t - measured.size.height).coerceAtLeast(0f)
                drawRect(Color(0xCC000000), topLeft = Offset(lx, ly), size = Size((measured.size.width + 4).toFloat(), measured.size.height.toFloat()))
                drawText(measured, topLeft = Offset(lx + 2f, ly))
            }

            // User-drawn missed-detection boxes (dashed blue, tap to remove)
            missedBoxes.forEach { box ->
                val l = ox + box.left * scale
                val t = oy + box.top * scale
                drawRect(
                    Color(0xFF8888FF),
                    topLeft = Offset(l, t),
                    size = Size(box.width * scale, box.height * scale),
                    style = Stroke(4f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 6f))),
                )
                val measured = textMeasurer.measure("FN (tap×)", TextStyle(color = Color.White, fontSize = 11.sp))
                val lx = l.coerceAtMost(sw - measured.size.width - 4f)
                val ly = (t - measured.size.height).coerceAtLeast(0f)
                drawRect(Color(0xCC000000), topLeft = Offset(lx, ly), size = Size((measured.size.width + 4).toFloat(), measured.size.height.toFloat()))
                drawText(measured, topLeft = Offset(lx + 2f, ly))
            }

            // In-progress box while dragging
            val ds = dragStart; val dc = dragCurrent
            if (drawingFnMode && ds != null && dc != null) {
                drawRect(
                    Color(0x998888FF),
                    topLeft = Offset(min(ds.x, dc.x), min(ds.y, dc.y)),
                    size = Size(abs(dc.x - ds.x), abs(dc.y - ds.y)),
                    style = Stroke(3f),
                )
            }
        }

        if (drawingFnMode) {
            Box(
                modifier = Modifier.fillMaxSize().pointerInput(canvasSize, frameWidth, frameHeight, zoomScale, panOffset) {
                    val sw = canvasSize.width.toFloat(); val sh = canvasSize.height.toFloat()
                    if (sw <= 0f || sh <= 0f) return@pointerInput
                    val scale = min(sw / frameWidth, sh / frameHeight)
                    val ox = (sw - frameWidth * scale) / 2f
                    val oy = (sh - frameHeight * scale) / 2f
                    val cx = sw / 2f; val cy = sh / 2f
                    // Inverse-transform a screen position (in graphicsLayer-transformed space)
                    // back to pre-transform layout space, then to frame coordinates.
                    fun toFrame(p: Offset): Offset {
                        val lx = if (zoomScale != 1f) cx + (p.x - panOffset.x - cx) / zoomScale else p.x
                        val ly = if (zoomScale != 1f) cy + (p.y - panOffset.y - cy) / zoomScale else p.y
                        return Offset((lx - ox) / scale, (ly - oy) / scale)
                    }
                    detectDragGestures(
                        onDragStart = { o -> dragStart = o; dragCurrent = o },
                        onDrag = { change, _ -> dragCurrent = change.position },
                        onDragEnd = {
                            val s = dragStart; val e = dragCurrent
                            if (s != null && e != null) {
                                val f1 = toFrame(s); val f2 = toFrame(e)
                                val fx1 = f1.x.coerceIn(0f, frameWidth.toFloat())
                                val fy1 = f1.y.coerceIn(0f, frameHeight.toFloat())
                                val fx2 = f2.x.coerceIn(0f, frameWidth.toFloat())
                                val fy2 = f2.y.coerceIn(0f, frameHeight.toFloat())
                                val box = BoundingBox(
                                    x = min(fx1, fx2), y = min(fy1, fy2),
                                    width = abs(fx2 - fx1), height = abs(fy2 - fy1),
                                )
                                if (box.width > 5f && box.height > 5f) onMissedBoxAdded(box)
                            }
                            dragStart = null; dragCurrent = null
                        },
                        onDragCancel = { dragStart = null; dragCurrent = null },
                    )
                },
            )
        } else {
            // Custom tap handler that observes without consuming the down event so
            // the page-level pinch-to-zoom still works. Events are only consumed when
            // the tap lands on a detection box. Touch coords are inverse-transformed
            // from graphicsLayer space back to pre-zoom layout space before hit testing.
            Box(
                modifier = Modifier.fillMaxSize().pointerInput(canvasSize, frameWidth, frameHeight, annotations, missedBoxes, zoomScale, panOffset) {
                    val sw = canvasSize.width.toFloat(); val sh = canvasSize.height.toFloat()
                    if (sw <= 0f || sh <= 0f) return@pointerInput
                    val scale = min(sw / frameWidth, sh / frameHeight)
                    val ox = (sw - frameWidth * scale) / 2f
                    val oy = (sh - frameHeight * scale) / 2f
                    val cx = sw / 2f; val cy = sh / 2f
                    val slop = viewConfiguration.touchSlop
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        var tapUp: PointerInputChange? = null
                        loop@ while (true) {
                            val event = awaitPointerEvent()
                            if (event.changes.count { it.pressed } > 1) break  // multi-touch → pinch
                            for (change in event.changes) {
                                if (change.id != down.id) continue
                                if (!change.pressed) {
                                    if ((change.position - down.position).getDistance() <= slop) tapUp = change
                                    break@loop
                                }
                                if ((change.position - down.position).getDistance() > slop) break@loop
                            }
                        }
                        val upChange = tapUp ?: return@awaitEachGesture
                        val raw = down.position
                        // Inverse-transform from graphicsLayer-transformed space to pre-zoom layout space
                        val lx = if (zoomScale != 1f) cx + (raw.x - panOffset.x - cx) / zoomScale else raw.x
                        val ly = if (zoomScale != 1f) cy + (raw.y - panOffset.y - cy) / zoomScale else raw.y
                        val fx = (lx - ox) / scale
                        val fy = (ly - oy) / scale
                        val missedHit = missedBoxes.indexOfFirst { b ->
                            fx >= b.left && fx <= b.right && fy >= b.top && fy <= b.bottom
                        }
                        if (missedHit >= 0) {
                            upChange.consume()
                            onMissedBoxRemoved(missedHit)
                            return@awaitEachGesture
                        }
                        val detHit = detections.indexOfFirst { d ->
                            fx >= d.box.left && fx <= d.box.right && fy >= d.box.top && fy <= d.box.bottom
                        }
                        if (detHit >= 0) {
                            upChange.consume()
                            val key = "$detHit"
                            val next = when (annotations[key]) {
                                null -> DetectionAnnotation.TRUE_POSITIVE
                                DetectionAnnotation.TRUE_POSITIVE -> DetectionAnnotation.FALSE_POSITIVE
                                DetectionAnnotation.FALSE_POSITIVE -> null
                            }
                            onAnnotationChanged(key, next)
                        }
                    }
                },
            )
        }
    }
}

@Composable
private fun PhotoPage(filePath: String) {
    var bitmap by remember(filePath) { mutableStateOf<ImageBitmap?>(null) }
    LaunchedEffect(filePath) {
        bitmap = withContext(Dispatchers.IO) { loadImageBitmapFromFile(filePath) }
    }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        val bmp = bitmap
        if (bmp != null) {
            Image(
                bitmap = bmp,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
        } else {
            CircularProgressIndicator(color = Color.White)
        }
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
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E1E30)),
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
