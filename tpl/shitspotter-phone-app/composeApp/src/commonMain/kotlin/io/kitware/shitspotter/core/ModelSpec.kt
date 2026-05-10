package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
enum class ModelFormat { ONNX, TFLITE, PTE, COREML, STUB }

@Serializable
enum class InputLayout { NCHW, NHWC }

@Serializable
enum class ColorOrder { RGB, BGR, YUV }

@Serializable
enum class ResizePolicy { LETTERBOX, STRETCH, CENTER_CROP }

@Serializable
enum class PostprocessType {
    YOLOX,
    YOLO_V9,
    /**
     * YOLOv9 multi-output export with per-stride class-logit tensors and
     * per-stride pre-decoded bbox (4 channels, left/top/right/bottom
     * distance in stride-units). Requires [ModelSpec.yolov9Schema] to be
     * non-null so the backend knows which output names to look at and
     * what stride each level represents.
     *
     * Class outputs are sigmoid-applied at postprocess. Bbox outputs are
     * multiplied by the level's stride to convert to pixel units.
     */
    YOLO_V9_DFL,
    GENERIC_BOX_SCORE_CLASS,
    NONE,
    STUB,
}

/**
 * Output-tensor schema for [PostprocessType.YOLO_V9_DFL]. The three
 * lists are parallel — the i-th class output / bbox output / stride
 * describe the same feature level (e.g. stride 8 = 80x80 grid for a
 * 640x640 input).
 */
@Serializable
data class Yolov9Schema(
    val classOutputs: List<String>,
    val bboxOutputs: List<String>,
    val strides: List<Int>,
) {
    init {
        require(classOutputs.size == bboxOutputs.size && classOutputs.size == strides.size) {
            "Yolov9Schema lists must align by length"
        }
    }
}

@Serializable
data class Normalization(
    val mean: List<Float> = listOf(0f, 0f, 0f),
    val std: List<Float> = listOf(1f, 1f, 1f),
    val scale: Float = 1f,
)

@Serializable
data class ModelSpec(
    val modelId: String,
    val displayName: String,
    val modelFile: String,
    val format: ModelFormat,
    val inputWidth: Int,
    val inputHeight: Int,
    val inputLayout: InputLayout,
    val colorOrder: ColorOrder,
    val normalization: Normalization,
    val resizePolicy: ResizePolicy,
    val postprocessType: PostprocessType,
    val classNames: List<String>,
    val scoreThreshold: Float = 0.25f,
    val iouThreshold: Float = 0.45f,
    val modelHash: String? = null,
    val modelVersion: String? = null,
    val trainingDatasetHint: String? = null,
    val notes: String? = null,
    /** Only used when [postprocessType] = YOLO_V9_DFL. */
    val yolov9Schema: Yolov9Schema? = null,
) {
    companion object {
        val STUB = ModelSpec(
            modelId = "stub-fake-detector",
            displayName = "Stub (no inference)",
            modelFile = "<none>",
            format = ModelFormat.STUB,
            inputWidth = 640,
            inputHeight = 640,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.STUB,
            classNames = listOf("poop"),
            scoreThreshold = 0.0f,
            notes = "Returns a deterministic fake box for Milestone 1 wiring tests.",
        )

        val YOLOX_NANO_POOP = ModelSpec(
            modelId = "yolox-nano-poop-cropped-v1",
            displayName = "YOLOX-nano poop (cropped, v1)",
            modelFile = "yolox_nano_poop_cropped_only_best.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 416,
            inputHeight = 416,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.YOLOX,
            classNames = listOf("poop"),
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
            notes = "First real model. Reference ONNX from tpl/poop_models/. " +
                "Input expected as raw RGB uint8 → float (no /255 in YOLOX preprocessing).",
        )

        /**
         * Larger custom-trained detector at 640x640, output shape
         * [1, 8400, 6] = [batch, anchors, (cx, cy, w, h, obj, cls0)]. Lives
         * at `tpl/poop_models/shitspotter-custom-v5-epoch_115.onnx`.
         * Uses the same YOLOX postprocess path as the nano model (decoded
         * format).
         */
        val CUSTOM_V5_EPOCH115 = ModelSpec(
            modelId = "shitspotter-custom-v5-epoch115",
            displayName = "Custom v5 epoch 115 (640x640)",
            modelFile = "shitspotter-custom-v5-epoch_115.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 640,
            inputHeight = 640,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.YOLOX,
            classNames = listOf("poop"),
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
            notes = "Larger custom detector. Output [1, 8400, 6]. " +
                "Pixel-normalised input (/ 255) — adjust if the export differs. " +
                "EMPIRICAL: at default thresholds this returns 0 detections on " +
                "out-of-distribution images like dog.jpg, where YOLOX-nano-poop " +
                "would over-predict. Confirm sigmoid vs raw-logit output before " +
                "trusting this on real device data.",
        )

        /**
         * YOLOv9-style anchor-free DFL detector trained on the
         * ShitSpotter dataset, simple-v3 run-v06 epoch 32. Multi-output
         * export with separate class-logit and pre-decoded bbox tensors
         * at strides 8/16/32. See `Yolov9.decode` for the math.
         *
         * Source file is the long-named DVC export; we use a shorter
         * `modelFile` so adb-pushing it stays tractable.
         */
        val SIMPLE_V3_RUN_V06 = ModelSpec(
            modelId = "shitspotter-simple-v3-run-v06",
            displayName = "Simple v3 run v06 (YOLOv9, 640x640)",
            modelFile = "shitspotter-simple-v3-run-v06.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 640,
            inputHeight = 640,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.YOLO_V9_DFL,
            classNames = listOf("poop"),
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
            yolov9Schema = Yolov9Schema(
                classOutputs = listOf("output", "2336", "2383"),
                bboxOutputs = listOf("2318", "2365", "2412"),
                strides = listOf(8, 16, 32),
            ),
            notes = "Multi-output YOLOv9 export. Class logits are raw " +
                "(sigmoid applied in postprocess); bbox is in stride-units " +
                "(multiplied by stride to get pixels). Aux head outputs " +
                "(2957/3004/3051/...) are ignored — only the main head is " +
                "used at inference time.",
        )

        /**
         * Earlier custom v2 detector. Same input/output shape as v5 but
         * trained on a smaller dataset. Useful as a comparison baseline.
         */
        val CUSTOM_V2_EPOCH126 = ModelSpec(
            modelId = "shitspotter-custom-v2-epoch126",
            displayName = "Custom v2 epoch 126 (640x640)",
            modelFile = "shitspotter_custom_v2_epoch126.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 640,
            inputHeight = 640,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.YOLOX,
            classNames = listOf("poop"),
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
            notes = "Earlier custom detector. Same shape as v5; useful as a " +
                "comparison baseline.",
        )
    }
}

object ModelRegistry {
    val all: List<ModelSpec> = listOf(
        ModelSpec.STUB,
        ModelSpec.YOLOX_NANO_POOP,
        ModelSpec.SIMPLE_V3_RUN_V06,
        ModelSpec.CUSTOM_V5_EPOCH115,
        ModelSpec.CUSTOM_V2_EPOCH126,
    )

    fun byId(id: String): ModelSpec? = all.firstOrNull { it.modelId == id }

    val default: ModelSpec = ModelSpec.STUB
}
