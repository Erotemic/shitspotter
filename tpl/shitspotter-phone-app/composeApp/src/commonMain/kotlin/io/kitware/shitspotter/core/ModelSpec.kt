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
    /**
     * DEIMv2 transformer detector with built-in postprocessor.
     *
     * Two inputs: "images" (float32 NCHW, normalised to [0,1]) and
     * "orig_target_sizes" (int64 [1,2] = [[W, H]] in model-input pixels).
     * Three outputs: "labels" (int64 [1,N]), "boxes" (float32 [1,N,4],
     * xyxy in orig_target_sizes pixel space), "scores" (float32 [1,N]).
     *
     * Requires [ModelSpec.deimv2Schema] to be non-null.
     */
    DEIMV2,
    GENERIC_BOX_SCORE_CLASS,
    NONE,
    STUB,
}

/**
 * I/O schema for [PostprocessType.DEIMV2]. Default values match the names
 * produced by the v4 ONNX export (03_export_onnx.sh).
 */
@Serializable
data class Deimv2Schema(
    val imagesInput: String = "images",
    val origSizeInput: String = "orig_target_sizes",
    val labelsOutput: String = "labels",
    val boxesOutput: String = "boxes",
    val scoresOutput: String = "scores",
)

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
    /** mAP@IoU=0.5 on the v9 simplified-GT test split, if known. */
    val apAt50: Float? = null,
    /** Human-readable speed hint shown in the model picker, e.g. "11 FPS (NNAPI)". */
    val fpsHint: String? = null,
    /** Only used when [postprocessType] = YOLO_V9_DFL. */
    val yolov9Schema: Yolov9Schema? = null,
    /** Only used when [postprocessType] = DEIMV2. */
    val deimv2Schema: Deimv2Schema? = null,
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
            classNames = listOf("stub"),
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

        // ---- v4 DEIMv2 sweep candidates (mobile_app_training_v4) ----------
        // All trained on the v9 tile-augmented split; ONNX files are in
        // $V4_ROOT/runs/<candidate_id>/export/ and should be pushed to the
        // device's models dir (see 05_bench_on_pixel5.sh / 07_register…).
        // Pixel 5 NNAPI latency measured 2026-05-14.

        val DEIMV2_PICO_320 = ModelSpec(
            modelId = "shitspotter-deimv2_pico-h320w320-fixed",
            displayName = "DEIMv2-Pico 320×320 (v4, 11 FPS)",
            modelFile = "deimv2_pico_h320_w320.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 320, inputHeight = 320,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.DEIMV2,
            classNames = listOf("poop"),
            scoreThreshold = 0.30f, iouThreshold = 0.45f,
            deimv2Schema = Deimv2Schema(),
            apAt50 = 0.265f,
            fpsHint = "11 FPS (NNAPI)",
            notes = "AP=0.265. Pixel 5 NNAPI: 89ms/11 FPS. Fastest v4 model.",
        )

        val DEIMV2_PICO_416 = ModelSpec(
            modelId = "shitspotter-deimv2_pico-h416w416-fixed",
            displayName = "DEIMv2-Pico 416×416 (v4, 7 FPS)",
            modelFile = "deimv2_pico_h416_w416.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 416, inputHeight = 416,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.DEIMV2,
            classNames = listOf("poop"),
            scoreThreshold = 0.30f, iouThreshold = 0.45f,
            deimv2Schema = Deimv2Schema(),
            apAt50 = 0.406f,
            fpsHint = "7 FPS (NNAPI)",
            notes = "AP=0.406. Pixel 5 NNAPI: 135ms/7 FPS.",
        )

        val DEIMV2_N_512 = ModelSpec(
            modelId = "shitspotter-deimv2_n-h512w512-fixed",
            displayName = "DEIMv2-N 512×512 (v4, 4 FPS)",
            modelFile = "deimv2_n_h512_w512.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 512, inputHeight = 512,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.DEIMV2,
            classNames = listOf("poop"),
            scoreThreshold = 0.30f, iouThreshold = 0.45f,
            deimv2Schema = Deimv2Schema(),
            apAt50 = 0.477f,
            fpsHint = "4 FPS (NNAPI)",
            notes = "AP=0.477. Pixel 5 NNAPI: 274ms/4 FPS.",
        )

        val DEIMV2_N_640 = ModelSpec(
            modelId = "shitspotter-deimv2_n-h640w640-fixed",
            displayName = "DEIMv2-N 640×640 (v4, 3 FPS)",
            modelFile = "deimv2_n_h640_w640.onnx",
            format = ModelFormat.ONNX,
            inputWidth = 640, inputHeight = 640,
            inputLayout = InputLayout.NCHW,
            colorOrder = ColorOrder.RGB,
            normalization = Normalization(scale = 1f / 255f),
            resizePolicy = ResizePolicy.LETTERBOX,
            postprocessType = PostprocessType.DEIMV2,
            classNames = listOf("poop"),
            scoreThreshold = 0.30f, iouThreshold = 0.45f,
            deimv2Schema = Deimv2Schema(),
            apAt50 = 0.520f,
            fpsHint = "3 FPS (NNAPI)",
            notes = "AP=0.520 (highest v4 AP). Pixel 5 NNAPI: 378ms/3 FPS.",
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
        // v4 DEIMv2 candidates — fastest first so the chip row reads left→right
        ModelSpec.DEIMV2_PICO_320,
        ModelSpec.DEIMV2_PICO_416,
        ModelSpec.DEIMV2_N_512,
        ModelSpec.DEIMV2_N_640,
        // legacy YOLOX / YOLOv9 baselines
        ModelSpec.YOLOX_NANO_POOP,
        ModelSpec.SIMPLE_V3_RUN_V06,
        ModelSpec.CUSTOM_V5_EPOCH115,
        ModelSpec.CUSTOM_V2_EPOCH126,
    )

    fun byId(id: String): ModelSpec? = all.firstOrNull { it.modelId == id }

    val default: ModelSpec = ModelSpec.STUB
}
