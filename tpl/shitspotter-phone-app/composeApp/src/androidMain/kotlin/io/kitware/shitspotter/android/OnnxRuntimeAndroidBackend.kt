package io.kitware.shitspotter.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import ai.onnxruntime.TensorInfo
import io.kitware.shitspotter.core.BACKEND_FLOOR_THRESHOLD
import io.kitware.shitspotter.core.BoundingBox
import io.kitware.shitspotter.core.ColorOrder
import io.kitware.shitspotter.core.Detection
import io.kitware.shitspotter.core.FeatureLevel
import io.kitware.shitspotter.core.Yolov9
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FrameSource
import io.kitware.shitspotter.core.InferenceResult
import io.kitware.shitspotter.core.InputLayout
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
import java.nio.LongBuffer
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
    private var imageInputName: String = ""
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
        imageInputName = spec.deimv2Schema?.imagesInput ?: s.inputNames.first()
        validateInputShape(s)
    }

    private fun validateInputShape(s: OrtSession) {
        val info = s.inputInfo[imageInputName] ?: return
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
        val imageTensor = OnnxTensor.createTensor(e, inputBuf, shape)
        val preprocessMs = nowMonoMs() - preStart

        var inferenceMs = 0.0
        var postprocessMs = 0.0
        val finalDets: List<Detection>
        try {
            val infStart = nowMonoMs()
            val inputs: Map<String, OnnxTensor> = when (spec.postprocessType) {
                PostprocessType.DEIMV2 -> {
                    val schema = spec.deimv2Schema!!
                    val origSizeBuf = LongBuffer.wrap(
                        longArrayOf(spec.inputWidth.toLong(), spec.inputHeight.toLong())
                    )
                    val origSizeTensor = OnnxTensor.createTensor(e, origSizeBuf, longArrayOf(1L, 2L))
                    mapOf(schema.imagesInput to imageTensor, schema.origSizeInput to origSizeTensor)
                }
                else -> mapOf(imageInputName to imageTensor)
            }
            val outputs = s.run(inputs)
            // origSizeTensor (if created) will be GC'd; ORT Android copies data eagerly.
            inferenceMs = nowMonoMs() - infStart
            try {
                val postStart = nowMonoMs()
                val numClasses = spec.classNames.size
                val rawDets: List<Detection> = when (spec.postprocessType) {
                    PostprocessType.DEIMV2 -> decodeDeimv2(outputs)
                    PostprocessType.YOLO_V9_DFL -> decodeYolov9(outputs, numClasses)
                    PostprocessType.YOLOX,
                    PostprocessType.YOLO_V9,
                    PostprocessType.GENERIC_BOX_SCORE_CLASS -> {
                        val raw = outputs[0]
                        val rawValue = raw.value
                        val flat = flattenOutput(rawValue)
                        val perRow = 5 + numClasses
                        require(flat.size % perRow == 0) {
                            "ONNX output shape ${flat.size} not divisible by $perRow"
                        }
                        val numAnchors = flat.size / perRow
                        Yolox.postprocessDecoded(
                            predictions = flat,
                            numAnchors = numAnchors,
                            numClasses = numClasses,
                            scoreThreshold = BACKEND_FLOOR_THRESHOLD,
                            iouThreshold = spec.iouThreshold,
                            classNames = spec.classNames,
                        )
                    }
                    PostprocessType.STUB, PostprocessType.NONE -> emptyList()
                }
                val mappedDets = rawDets.map { d ->
                    d.copy(box = lbParams.mapBoxToSource(d.box))
                }
                finalDets = if (mappedDets.size > 1)
                    Nms.apply(mappedDets, spec.iouThreshold) else mappedDets
                postprocessMs = nowMonoMs() - postStart
            } finally {
                outputs.close()
            }
        } finally {
            imageTensor.close()
        }

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

    private fun decodeDeimv2(outputs: OrtSession.Result): List<Detection> {
        val schema = spec.deimv2Schema
            ?: error("DEIMV2 requires ModelSpec.deimv2Schema; '${spec.modelId}' has none")
        val boxes  = flattenOutput(outputs.get(schema.boxesOutput).orElseThrow().value)
        val scores = flattenOutput(outputs.get(schema.scoresOutput).orElseThrow().value)
        val n = scores.size
        val dets = ArrayList<Detection>(n / 10)
        for (i in 0 until n) {
            val score = scores[i]
            if (score < BACKEND_FLOOR_THRESHOLD) continue
            val x0 = boxes[i * 4]
            val y0 = boxes[i * 4 + 1]
            val x1 = boxes[i * 4 + 2]
            val y1 = boxes[i * 4 + 3]
            dets += Detection(
                box = BoundingBox(x0, y0, x1 - x0, y1 - y0),
                score = score,
                classId = 0,
                className = spec.classNames.getOrElse(0) { "poop" },
            )
        }
        return dets
    }

    private fun decodeYolov9(
        outputs: OrtSession.Result,
        numClasses: Int,
    ): List<Detection> {
        val schema = spec.yolov9Schema
            ?: error("YOLO_V9_DFL requires ModelSpec.yolov9Schema; '${spec.modelId}' has none")
        val levels = ArrayList<FeatureLevel>(schema.classOutputs.size)
        for (i in schema.classOutputs.indices) {
            val classOut = outputs.get(schema.classOutputs[i]).orElseThrow {
                error("YOLO_V9_DFL: missing output '${schema.classOutputs[i]}' on '${spec.modelId}'")
            }
            val bboxOut = outputs.get(schema.bboxOutputs[i]).orElseThrow {
                error("YOLO_V9_DFL: missing output '${schema.bboxOutputs[i]}' on '${spec.modelId}'")
            }
            val classShape = (classOut.info as TensorInfo).shape
            val gridH = classShape[classShape.size - 2].toInt()
            val gridW = classShape[classShape.size - 1].toInt()
            levels += FeatureLevel(
                classLogits = flattenOutput(classOut.value),
                bbox = flattenOutput(bboxOut.value),
                gridH = gridH,
                gridW = gridW,
                stride = schema.strides[i],
            )
        }
        return Yolov9.decode(
            levels = levels,
            numClasses = numClasses,
            scoreThreshold = BACKEND_FLOOR_THRESHOLD,
            iouThreshold = spec.iouThreshold,
            classNames = spec.classNames,
        )
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
