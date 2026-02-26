package com.example.ua_money

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.ua_money.ui.theme.UA_MoneyTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var tts: TextToSpeech

    private val INPUT_SIZE = 1280
    private val CONFIDENCE_THRESHOLD = 0.5f
    private val IOU_THRESHOLD = 0.25f
    private val NUM_CLASSES = 10

    companion object {
        private const val TAG = "UA_Money"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            Log.d(TAG, "🔄 Початок завантаження моделі...")

            // Перевіряємо, чи файл існує
            val assetManager = assets
            val modelPath = "best_float32.tflite"

            try {
                val fileList = assetManager.list("") ?: emptyArray()
                Log.d(TAG, "📁 Файли в assets: ${fileList.joinToString()}")

                if (!fileList.contains(modelPath)) {
                    Log.e(TAG, "❌ Файл $modelPath НЕ ЗНАЙДЕНО в assets!")
                    Toast.makeText(this, "Модель не знайдена! Перевірте папку assets", Toast.LENGTH_LONG).show()
                    return
                }
            } catch (e: Exception) {
                Log.e(TAG, "❌ Помилка читання assets: ${e.message}")
            }

            val options = Interpreter.Options()
            options.setNumThreads(4)

            Log.d(TAG, "🔄 Завантажуємо модель з assets/$modelPath...")
            val modelBuffer = FileUtil.loadMappedFile(this, modelPath)
            Log.d(TAG, "✅ MappedByteBuffer створено, розмір: ${modelBuffer.capacity()} байт")

            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "✅ Interpreter створено")

            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            Log.d(TAG, "✅ Модель завантажена успішно!")
            Log.d(TAG, "📥 Вхід: ${inputTensor.shape().contentToString()} (${inputTensor.dataType()})")
            Log.d(TAG, "📤 Вихід: ${outputTensor.shape().contentToString()} (${outputTensor.dataType()})")

        } catch (e: Exception) {
            e.printStackTrace()
            Log.e(TAG, "❌ КРИТИЧНА ПОМИЛКА завантаження моделі!")
            Log.e(TAG, "Тип помилки: ${e.javaClass.simpleName}")
            Log.e(TAG, "Повідомлення: ${e.message}")
            Log.e(TAG, "Stack trace:")
            e.stackTrace.forEach { Log.e(TAG, "  $it") }

            Toast.makeText(
                this,
                "ПОМИЛКА: ${e.javaClass.simpleName}\n${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale("uk", "UA")
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 0)
        }

        setContent {
            UA_MoneyTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    MainScreen()
                }
            }
        }
    }

    data class Detection(val classId: Int, val score: Float, val rect: RectF)

    data class ProcessingResult(
        val total: Float,
        val detections: List<Detection>,
        val annotatedImage: Bitmap?
    )

    private fun preprocessImage(bitmap: Bitmap): Bitmap {
        try {
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            var minBrightness = 255
            var maxBrightness = 0

            for (pixel in pixels) {
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                val brightness = (r + g + b) / 3

                if (brightness < minBrightness) minBrightness = brightness
                if (brightness > maxBrightness) maxBrightness = brightness
            }

            val range = maxBrightness - minBrightness
            if (range > 50) {
                for (i in pixels.indices) {
                    val pixel = pixels[i]
                    var r = ((pixel shr 16) and 0xFF)
                    var g = ((pixel shr 8) and 0xFF)
                    var b = (pixel and 0xFF)

                    r = ((r - minBrightness) * 255 / range).coerceIn(0, 255)
                    g = ((g - minBrightness) * 255 / range).coerceIn(0, 255)
                    b = ((b - minBrightness) * 255 / range).coerceIn(0, 255)

                    pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            result.setPixels(pixels, 0, width, 0, 0, width, height)

            Log.d(TAG, "✅ Preprocessing: brightness range $minBrightness-$maxBrightness")
            return result

        } catch (e: Exception) {
            Log.e(TAG, "⚠️ Preprocessing failed, using original: ${e.message}")
            return bitmap
        }
    }

    private fun resizeWithPadding(bitmap: Bitmap, targetSize: Int): Pair<Bitmap, FloatArray> {
        val origWidth = bitmap.width
        val origHeight = bitmap.height

        val scale = targetSize.toFloat() / max(origWidth, origHeight)
        val scaledWidth = (origWidth * scale).toInt()
        val scaledHeight = (origHeight * scale).toInt()

        Log.d(TAG, "📐 Масштабування: ${origWidth}x${origHeight} -> ${scaledWidth}x${scaledHeight}")

        val scaled = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val padX = (targetSize - scaledWidth) / 2f
        val padY = (targetSize - scaledHeight) / 2f

        canvas.drawBitmap(scaled, padX, padY, null)
        scaled.recycle()

        val params = floatArrayOf(scale, padX, padY)
        return Pair(result, params)
    }

    // Малює bounding boxes на оригінальному зображенні
    private fun drawBoundingBoxes(
        originalBitmap: Bitmap,
        detections: List<Detection>
    ): Bitmap {
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 8f
            isAntiAlias = true
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
            isAntiAlias = true
            style = Paint.Style.FILL
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }

        val backgroundPaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Кольори для різних номіналів
        val colors = listOf(
            Color.rgb(255, 0, 0),      // Червоний
            Color.rgb(0, 255, 0),      // Зелений
            Color.rgb(0, 0, 255),      // Синій
            Color.rgb(255, 255, 0),    // Жовтий
            Color.rgb(255, 0, 255),    // Пурпурний
            Color.rgb(0, 255, 255),    // Блакитний
            Color.rgb(255, 165, 0),    // Помаранчевий
            Color.rgb(128, 0, 128),    // Фіолетовий
            Color.rgb(0, 128, 128),    // Темно-бірюзовий
            Color.rgb(255, 192, 203)   // Рожевий
        )

        val width = originalBitmap.width
        val height = originalBitmap.height

        detections.forEach { detection ->
            // Конвертуємо нормалізовані координати назад у пікселі
            val left = detection.rect.left * width
            val top = detection.rect.top * height
            val right = detection.rect.right * width
            val bottom = detection.rect.bottom * height

            // Вибираємо колір для класу
            val color = colors[detection.classId % colors.size]
            boxPaint.color = color
            backgroundPaint.color = color

            // Малюємо bounding box
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Підготовка тексту
            val label = "${getClassName(detection.classId)} ${(detection.score * 100).toInt()}%"
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)

            // Позиція тексту (над боксом)
            val textX = left + 10f
            val textY = max(top - 10f, textBounds.height() + 10f)

            // Фон для тексту (напівпрозорий)
            backgroundPaint.alpha = 180
            canvas.drawRect(
                textX - 5f,
                textY - textBounds.height() - 5f,
                textX + textBounds.width() + 10f,
                textY + 5f,
                backgroundPaint
            )

            // Малюємо текст
            canvas.drawText(label, textX, textY, textPaint)
        }

        Log.d(TAG, "🎨 Намальовано ${detections.size} bounding boxes")
        return mutableBitmap
    }

    private suspend fun processImage(originalBitmap: Bitmap): ProcessingResult = withContext(Dispatchers.Default) {
        try {
            if (!::interpreter.isInitialized) {
                Log.e(TAG, "❌ Інтерпретатор не ініціалізовано")
                return@withContext ProcessingResult(0f, emptyList(), null)
            }

            Log.d(TAG, "📸 Оригінальне зображення: ${originalBitmap.width}x${originalBitmap.height}")

            val preprocessed = preprocessImage(originalBitmap)
            val (resizedBitmap, transformParams) = resizeWithPadding(preprocessed, INPUT_SIZE)
            val (scale, padX, padY) = transformParams

            if (preprocessed != originalBitmap) preprocessed.recycle()

            saveDebugImage(resizedBitmap)

            val inputBuffer = java.nio.ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
            inputBuffer.order(java.nio.ByteOrder.nativeOrder())

            val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
            resizedBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

            for (pixelValue in intValues) {
                inputBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((pixelValue and 0xFF) / 255f)
            }
            inputBuffer.rewind()
            resizedBitmap.recycle()

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            Log.d(TAG, "🔢 Форма виходу: ${outputShape.contentToString()}")

            val outputArray = Array(outputShape[0]) {
                Array(outputShape[1]) {
                    FloatArray(outputShape[2])
                }
            }

            Log.d(TAG, "🚀 Запуск інференсу...")
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputArray)
            val inferenceTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "✅ Інференс завершено за ${inferenceTime}мс")

            // Парсимо детекції (координати в межах INPUT_SIZE з padding)
            val rawDetections = parseDetections(outputArray, outputShape)

            // Конвертуємо координати назад до оригінального зображення
            val adjustedDetections = rawDetections.map { detection ->
                // Координати зараз нормалізовані (0-1) відносно INPUT_SIZE
                // Конвертуємо до координат оригінального зображення

                // Спочатку перетворюємо з нормалізованих до пікселів INPUT_SIZE
                var left = detection.rect.left * INPUT_SIZE
                var top = detection.rect.top * INPUT_SIZE
                var right = detection.rect.right * INPUT_SIZE
                var bottom = detection.rect.bottom * INPUT_SIZE

                // Віднімаємо padding
                left -= padX
                top -= padY
                right -= padX
                bottom -= padY

                // Перетворюємо назад з масштабованих координат
                left /= scale
                top /= scale
                right /= scale
                bottom /= scale

                // Нормалізуємо до розміру оригінального зображення
                val normLeft = left / originalBitmap.width
                val normTop = top / originalBitmap.height
                val normRight = right / originalBitmap.width
                val normBottom = bottom / originalBitmap.height

                Detection(
                    detection.classId,
                    detection.score,
                    RectF(
                        normLeft.coerceIn(0f, 1f),
                        normTop.coerceIn(0f, 1f),
                        normRight.coerceIn(0f, 1f),
                        normBottom.coerceIn(0f, 1f)
                    )
                )
            }

            val finalDetections = applyNMS(adjustedDetections)
            Log.d(TAG, "🎯 Після NMS: ${finalDetections.size}")

            finalDetections.forEach {
                Log.d(TAG, "💰 Фінальна: ${getClassName(it.classId)} - впевненість: ${it.score}")
            }

            val total = calculateTotal(finalDetections)
            Log.d(TAG, "💵 Загальна сума: $total грн")

            // Малюємо bounding boxes на оригінальному зображенні
            val annotatedImage = if (finalDetections.isNotEmpty()) {
                drawBoundingBoxes(originalBitmap, finalDetections)
            } else {
                null
            }

            return@withContext ProcessingResult(total, finalDetections, annotatedImage)

        } catch (e: Exception) {
            e.printStackTrace()
            Log.e(TAG, "❌ Помилка обробки: ${e.message}")
            Log.e(TAG, "Stack trace: ${e.stackTraceToString()}")
            withContext(Dispatchers.Main) {
                Toast.makeText(this@MainActivity, "Помилка: ${e.message}", Toast.LENGTH_SHORT).show()
            }
            return@withContext ProcessingResult(0f, emptyList(), null)
        }
    }

    private fun parseDetections(outputArray: Array<Array<FloatArray>>, outputShape: IntArray): List<Detection> {
        val rawDetections = ArrayList<Detection>()
        val numBoxes = outputShape[2]

        Log.d(TAG, "📊 Обробка $numBoxes боксів...")

        for (i in 0 until min(5, numBoxes)) {
            val cx = outputArray[0][0][i]
            val cy = outputArray[0][1][i]
            val w = outputArray[0][2][i]
            val h = outputArray[0][3][i]
            val scores = (0 until NUM_CLASSES).map { outputArray[0][4 + it][i] }
            Log.d(TAG, "RAW бокс $i: cx=$cx, cy=$cy, w=$w, h=$h, scores=$scores")
        }

        var detectedAboveThreshold = 0

        for (i in 0 until numBoxes) {
            val cx = outputArray[0][0][i]
            val cy = outputArray[0][1][i]
            val w = outputArray[0][2][i]
            val h = outputArray[0][3][i]

            var maxScore = 0f
            var maxClassIndex = -1
            for (c in 0 until NUM_CLASSES) {
                val score = outputArray[0][4 + c][i]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIndex = c
                }
            }

            if (maxScore > CONFIDENCE_THRESHOLD) {
                detectedAboveThreshold++

                val isNormalized = cx <= 1f && cy <= 1f && w <= 1f && h <= 1f

                val left: Float
                val top: Float
                val right: Float
                val bottom: Float

                if (isNormalized) {
                    left = cx - w / 2
                    top = cy - h / 2
                    right = cx + w / 2
                    bottom = cy + h / 2
                } else {
                    left = (cx - w / 2) / INPUT_SIZE
                    top = (cy - h / 2) / INPUT_SIZE
                    right = (cx + w / 2) / INPUT_SIZE
                    bottom = (cy + h / 2) / INPUT_SIZE
                }

                if (left >= -0.1f && top >= -0.1f && right <= 1.1f && bottom <= 1.1f &&
                    right > left && bottom > top) {

                    val clampedRect = RectF(
                        left.coerceIn(0f, 1f),
                        top.coerceIn(0f, 1f),
                        right.coerceIn(0f, 1f),
                        bottom.coerceIn(0f, 1f)
                    )

                    rawDetections.add(Detection(maxClassIndex, maxScore, clampedRect))

                    if (i < 10) {
                        Log.d(TAG, "✅ Детекція #$i: ${getClassName(maxClassIndex)}, " +
                                "score=$maxScore, rect=$clampedRect")
                    }
                } else {
                    if (i < 10) {
                        Log.d(TAG, "❌ Відкинуто бокс #$i: некоректні координати " +
                                "left=$left, top=$top, right=$right, bottom=$bottom")
                    }
                }
            }
        }

        Log.d(TAG, "📊 Детекцій з порогом > $CONFIDENCE_THRESHOLD: $detectedAboveThreshold")
        Log.d(TAG, "📦 Валідних детекцій: ${rawDetections.size}")

        return rawDetections
    }

    private fun getClassName(classId: Int): String {
        return when(classId) {
            0 -> "1 грн"
            1 -> "10 грн"
            2 -> "100 грн"
            3 -> "1000 грн"
            4 -> "2 грн"
            5 -> "20 грн"
            6 -> "200 грн"
            7 -> "5 грн"
            8 -> "50 грн"
            9 -> "500 грн"
            else -> "Невідомо ($classId)"
        }
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val selected = ArrayList<Detection>()

        while (sorted.isNotEmpty()) {
            val current = sorted.removeAt(0)
            selected.add(current)

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                val iou = calculateIoU(current.rect, next.rect)

                if (current.classId == next.classId && iou > IOU_THRESHOLD) {
                    Log.d(TAG, "🗑️ NMS видалив: ${getClassName(next.classId)}, IoU=$iou")
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersectionLeft = max(a.left, b.left)
        val intersectionTop = max(a.top, b.top)
        val intersectionRight = min(a.right, b.right)
        val intersectionBottom = min(a.bottom, b.bottom)

        if (intersectionRight < intersectionLeft || intersectionBottom < intersectionTop) return 0f

        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val unionArea = areaA + areaB - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    private fun calculateTotal(detections: List<Detection>): Float {
        val classToValue = mapOf(
            0 to 1f, 1 to 10f, 2 to 100f, 3 to 1000f, 4 to 2f,
            5 to 20f, 6 to 200f, 7 to 5f, 8 to 50f, 9 to 500f
        )
        return detections.sumOf { (classToValue[it.classId] ?: 0f).toDouble() }.toFloat()
    }

    private fun saveDebugImage(bitmap: Bitmap) {
        try {
            val file = File(cacheDir, "debug_processed.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            Log.d(TAG, "💾 Debug image saved: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "⚠️ Failed to save debug image: ${e.message}")
        }
    }

    @Composable
    fun MainScreen() {
        val context = LocalContext.current
        var result by remember { mutableStateOf<ProcessingResult?>(null) }
        var isProcessing by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()
        val scrollState = rememberScrollState()

        val photoFile = remember { File(context.cacheDir, "temp_photo.jpg") }
        val photoUri = remember {
            FileProvider.getUriForFile(context, "${context.packageName}.provider", photoFile)
        }

        val launcher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                isProcessing = true
                scope.launch {
                    val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    if (bitmap != null) {
                        Log.d(TAG, "✅ Фото завантажено: ${bitmap.width}x${bitmap.height}")
                        val processingResult = processImage(bitmap)
                        result = processingResult
                        bitmap.recycle()

                        withContext(Dispatchers.Main) {
                            if (processingResult.total > 0) {
                                tts.speak("Сума: ${processingResult.total.toInt()} гривень",
                                    TextToSpeech.QUEUE_FLUSH, null, null)
                            } else {
                                tts.speak("Купюри не знайдено",
                                    TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                        }
                    } else {
                        Log.e(TAG, "❌ Не вдалося завантажити фото")
                        withContext(Dispatchers.Main) {
                            Toast.makeText(context, "Помилка читання фото", Toast.LENGTH_SHORT).show()
                        }
                    }
                    isProcessing = false
                }
            } else {
                Log.w(TAG, "⚠️ Фото скасовано")
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(scrollState),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (isProcessing) {
                Spacer(modifier = Modifier.height(48.dp))
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Аналізую фото...", style = MaterialTheme.typography.bodyLarge)
            } else {
                result?.let { res ->
                    // Показуємо зображення з bounding boxes
                    res.annotatedImage?.let { img ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                        ) {
                            Image(
                                bitmap = img.asImageBitmap(),
                                contentDescription = "Розпізнані купюри",
                                modifier = Modifier.fillMaxWidth(),
                                contentScale = ContentScale.FillWidth
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Показуємо результати
                    if (res.total == 0f) {
                        Text(
                            text = "Купюри не знайдено",
                            style = MaterialTheme.typography.headlineMedium,
                            color = MaterialTheme.colorScheme.error
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Поради:\n" +
                                    "• Яскраве рівномірне освітлення\n" +
                                    "• Купюри на контрастному фоні\n" +
                                    "• Відстань 20-40 см\n" +
                                    "• Уникайте тіней і відблисків\n" +
                                    "• Купюри не повинні перекриватися",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    } else {
                        Text(
                            text = "💰 ${res.total.toInt()} грн",
                            style = MaterialTheme.typography.displayLarge,
                            color = MaterialTheme.colorScheme.primary
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        // Детальний список знайдених купюр
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text(
                                    text = "Знайдено купюр: ${res.detections.size}",
                                    style = MaterialTheme.typography.titleMedium
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                res.detections.forEach { detection ->
                                    Text(
                                        text = "• ${getClassName(detection.classId)} - " +
                                                "${(detection.score * 100).toInt()}%",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                }
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                }

                Button(
                    onClick = {
                        Log.d(TAG, "📷 Запуск камери...")
                        launcher.launch(photoUri)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp)
                ) {
                    Text("📷 Зробити фото", style = MaterialTheme.typography.titleMedium)
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Поріг: $CONFIDENCE_THRESHOLD | IoU: $IOU_THRESHOLD",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::tts.isInitialized) tts.shutdown()
        if (::interpreter.isInitialized) interpreter.close()
        Log.d(TAG, "🔚 Activity знищено")
    }
}