package io.kitware.shitspotter.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import io.kitware.shitspotter.core.BoundingBox
import io.kitware.shitspotter.core.ColorOrder
import io.kitware.shitspotter.core.Detection
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FrameSource
import io.kitware.shitspotter.core.InferenceResult
import io.kitware.shitspotter.core.InputLayout
import io.kitware.shitspotter.core.LetterboxParams
import io.kitware.shitspotter.core.ModelSpec
import io.kitware.shitspotter.core.Nms
import io.kitware.shitspotter.core.Normalization
import io.kitware.shitspotter.core.PostprocessType
import io.kitware.shitspotter.core.Preprocessing
import io.kitware.shitspotter.core.ResizePolicy
import io.kitware.shitspotter.core.Yolox
import io.kitware.shitspotter.core.nowMonoMs
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet

/**
 * ONNX Runtime Android backend. Tries NNAPI execution provider first; if that
 * fails (e.g. unsupported ops on Pixel 5 Adreno), falls back to CPU. The
 * actual delegate that succeeded is recorded in [InferenceResult.delegate]
 * so the HUD/telemetry never lies about acceleration.
 */
class OnnxRuntimeAndroidBackend(
    override val spec: ModelSpec,
    private val modelPath: String,
    private val tryNnapi: Boolean = true,
) : DetectorBackend {
    override val backendName: String = "onnxruntime-android-1.19"
    override var delegate: String? = null
        private set

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String = ""
    private var closed = false

    init {
        require(File(modelPath).isFile) { "ONNX model missing: $modelPath" }
        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        var lastErr: Throwable? = null
        if (tryNnapi) {
            try {
                opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                session = env!!.createSession(modelPath, opts)
                delegate = "NNAPI"
            } catch (t: Throwable) {
                lastErr = t
                session = null
            }
        }
        if (session == null) {
            try {
                val cpuOpts = OrtSession.SessionOptions()
                cpuOpts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                session = env!!.createSession(modelPath, cpuOpts)
                delegate = "CPU"
            } catch (t: Throwable) {
                lastErr = t
            }
        }
        val s = session ?: throw IllegalStateException(
            "Could not create ONNX session for $modelPath", lastErr
        )
        inputName = s.inputNames.first()
        validateInputShape(s)
    }

    private fun validateInputShape(s: OrtSession) {
        val info = s.inputInfo[inputName] ?: return
        val ti = info.info as? ai.onnxruntime.TensorInfo ?: return
        val shape = ti.shape
        if (shape.size != 4) return
        val (modelH, modelW) = when (spec.inputLayout) {
            InputLayout.NCHW -> shape[2].toInt() to shape[3].toInt()
            InputLayout.NHWC -> shape[1].toInt() to shape[2].toInt()
        }
        if (modelH > 0 && modelH != spec.inputHeight) {
            error("ONNX expects H=$modelH but ${spec.modelId} declares ${spec.inputHeight}")
        }
        if (modelW > 0 && modelW != spec.inputWidth) {
            error("ONNX expects W=$modelW but ${spec.modelId} declares ${spec.inputWidth}")
        }
    }

    override fun warmup() {
        val dummy = ByteArray(spec.inputWidth * spec.inputHeight * 3)
        analyze(object : FrameSource {
            override val width = spec.inputWidth
            override val height = spec.inputHeight
            override val rotationDegrees = 0
            override fun toRgb888(): ByteArray = dummy
        })
    }

    override fun analyze(frame: FrameSource): InferenceResult {
        check(!closed) { "Backend already closed" }
        val s = session ?: error("Session not initialized")
        val e = env ?: error("Env not initialized")

        val preStart = nowMonoMs()
        val rgb = frame.toRgb888()
        val (lbBytes, lbParams) = when (spec.resizePolicy) {
            ResizePolicy.LETTERBOX ->
                Preprocessing.letterboxRgb(rgb, frame.width, frame.height, spec.inputWidth, spec.inputHeight)
            ResizePolicy.STRETCH ->
                Preprocessing.stretchRgb(rgb, frame.width, frame.height, spec.inputWidth, spec.inputHeight)
            ResizePolicy.CENTER_CROP ->
                Preprocessing.centerCropRgb(rgb, frame.width, frame.height, spec.inputWidth, spec.inputHeight)
        }
        val tensorData = Preprocessing.toFloatTensor(
            rgb = lbBytes,
            width = spec.inputWidth,
            height = spec.inputHeight,
            layout = spec.inputLayout,
            colorOrder = spec.colorOrder,
            normalization = spec.normalization,
        )
        val shape: LongArray = when (spec.inputLayout) {
            InputLayout.NCHW -> longArrayOf(1L, 3L, spec.inputHeight.toLong(), spec.inputWidth.toLong())
            InputLayout.NHWC -> longArrayOf(1L, spec.inputHeight.toLong(), spec.inputWidth.toLong(), 3L)
        }
        val inputBuf = FloatBuffer.wrap(tensorData)
        val inputTensor = OnnxTensor.createTensor(e, inputBuf, shape)
        val preprocessMs = nowMonoMs() - preStart

        val infStart = nowMonoMs()
        val outputs = s.run(mapOf(inputName to inputTensor))
        val inferenceMs = nowMonoMs() - infStart

        val postStart = nowMonoMs()
        val raw = outputs[0]
        val rawValue = raw.value
        val flat = flattenOutput(rawValue)
        val numClasses = spec.classNames.size
        val perRow = 5 + numClasses
        require(flat.size % perRow == 0) {
            "ONNX output shape ${flat.size} not divisible by $perRow"
        }
        val numAnchors = flat.size / perRow
        val rawDets = when (spec.postprocessType) {
            PostprocessType.YOLOX -> Yolox.postprocessDecoded(
                predictions = flat,
                numAnchors = numAnchors,
                numClasses = numClasses,
                scoreThreshold = spec.scoreThreshold,
                iouThreshold = spec.iouThreshold,
                classNames = spec.classNames,
            )
            PostprocessType.YOLO_V9, PostprocessType.GENERIC_BOX_SCORE_CLASS ->
                Yolox.postprocessDecoded(
                    predictions = flat,
                    numAnchors = numAnchors,
                    numClasses = numClasses,
                    scoreThreshold = spec.scoreThreshold,
                    iouThreshold = spec.iouThreshold,
                    classNames = spec.classNames,
                )
            PostprocessType.STUB, PostprocessType.NONE -> emptyList()
        }
        val mappedDets = rawDets.map { d ->
            d.copy(box = lbParams.mapBoxToSource(d.box))
        }
        val finalDets = if (mappedDets.size > 1)
            Nms.apply(mappedDets, spec.iouThreshold) else mappedDets

        outputs.close()
        inputTensor.close()
        val postprocessMs = nowMonoMs() - postStart

        return InferenceResult(
            detections = finalDets,
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = postprocessMs,
            backendName = backendName,
            delegate = delegate,
        )
    }

    override fun close() {
        if (closed) return
        closed = true
        try { session?.close() } catch (_: Throwable) {}
        session = null
        env = null
    }

    private fun flattenOutput(v: Any?): FloatArray {
        return when (v) {
            is FloatArray -> v
            is Array<*> -> {
                val flat = ArrayList<Float>()
                flattenInto(v, flat)
                FloatArray(flat.size) { flat[it] }
            }
            else -> error("Unsupported ONNX output type: ${v?.let { it::class }}")
        }
    }

    private fun flattenInto(a: Array<*>, out: ArrayList<Float>) {
        for (e in a) {
            when (e) {
                is FloatArray -> for (f in e) out.add(f)
                is Array<*> -> flattenInto(e, out)
                is Float -> out.add(e)
                else -> error("Unsupported nested ONNX output element: ${e?.let { it::class }}")
            }
        }
    }
}
