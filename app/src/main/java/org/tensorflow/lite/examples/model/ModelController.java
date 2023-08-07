package org.tensorflow.lite.examples.model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.example.tf_lite_ex.R;

public class ModelController {
    private Interpreter interpreter = null;
    private int numThreads = 2;
    private Context context;
    private static String TAG = "TF_LITE_LOG";
    private int targetHeight = 0;
    private int targetWidth = 0;
    private class TrainingSample{
        TensorImage image;
        int label;

        public TrainingSample(TensorImage tensorImage, int encoding) {
            image = tensorImage;
            label = encoding;
        }
    }
    private ArrayList<TrainingSample> trainingSamples = new ArrayList<>();
    private String YES = "YES";
    private String NO = "NO";


    public ModelController(Context context){
        this.context = context;
        if( setupModel(context)) {
            targetHeight = interpreter.getInputTensor(0).shape()[2];
            targetWidth = interpreter.getInputTensor(0).shape()[1];

            Log.i(TAG, "WIDTH " + targetWidth + "HEIght " + targetHeight);
        }
    }

    public boolean setupModel(Context context) {
        Log.i(TAG, "SETUP MODEL...");
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(numThreads);
            MappedByteBuffer modelFile = FileUtil.loadMappedFile(context, "model.tflite");
            interpreter = new Interpreter(modelFile, options);
            Log.i(TAG, "Load model successfully!");
            return true;
        }catch (IOException e) {
            Log.e(TAG, "TFlite failed to load model with error: " + e.getMessage());
            return false;
        }
    }

    public void startTraining() {
        Log.i(TAG, "START TRAINING...");
        int NUM_EPOCHS = 100;
        int BATCH_SIZE = 4;
        int IMG_HEIGHT = 160;
        int IMG_WIDTH = 160;
        int NUM_BATCHES = (int)Math.ceil((double)trainingSamples.size() / BATCH_SIZE);
        trainingSamples.clear();

        String noFolderPath = context.getExternalFilesDir(null) + "/brain/" + NO;
        String yesFolderPath = context.getExternalFilesDir(null) + "/brain/" + YES;

        File noImageFolder = new File(noFolderPath);
        File yesImageFolder = new File(yesFolderPath);
        yesImageFolder.listFiles((dir, name) -> {
            Bitmap bitmap = BitmapFactory.decodeFile(dir.getAbsolutePath() + "/" + name);
            addSample(bitmap, YES);
            return true;
        });
        noImageFolder.listFiles((dir, name) -> {
            Bitmap bitmap = BitmapFactory.decodeFile(dir.getAbsolutePath() + "/" + name);
            addSample(bitmap, NO);
            return true;
        });

        Log.d(TAG, "NUM SAMPLES: " + String.valueOf(trainingSamples.size()));



        float[] losses = new float[NUM_EPOCHS];

        int[] inputShape = interpreter.getInputTensor(0).shape();

        for(int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            Collections.shuffle(trainingSamples);
            Iterator<List<TrainingSample>> batchIterator = trainingBatches(BATCH_SIZE);
            int batchIdx = 0;
            FloatBuffer loss = FloatBuffer.allocate(1);
            while (batchIterator.hasNext()){
                batchIdx +=1;
                List<TrainingSample> batch = batchIterator.next();
                float[][][][] trainImages = new float[BATCH_SIZE][targetHeight][targetWidth][3];
                float[] trainLabels = new float[BATCH_SIZE];
                //fill value
                int index = 0;
                for (TrainingSample sample: batch) {

                    float[] imageData = sample.image.getTensorBuffer().getFloatArray();
                    for (int y = 0; y < targetHeight; y++) {
                        for (int x = 0; x < targetWidth; x++) {
                            for (int c = 0; c < 3; c++) {
                                trainImages[index][y][x][c] = imageData[y * targetWidth * 3 + x * 3 + c];
                            }
                        }
                    }


                    float labelData = sample.label;
                    trainLabels[index] = labelData;
                    index ++;
                }

                Map<String, Object> inputs = new HashMap<>();
                inputs.put("x", trainImages);
                inputs.put("y", trainLabels);

                Map<String, Object> outputs = new HashMap<>();
                loss.rewind();
                outputs.put("loss", loss);

                interpreter.runSignature(inputs, outputs, "train");

                if(batchIdx == NUM_BATCHES -1 ) losses[epoch] = loss.get(0);


            }
            if ((epoch + 1) %10 == 0) {
                Log.i(TAG,
                        "Finished" + (epoch + 1) + "/" + NUM_EPOCHS + ", current loss: " + loss.get(0));
            }
        }


    }

    public void addSample(Bitmap image, String className) {
        if(interpreter== null) {
            setupModel(context);
        }
        TensorImage tensorImage = preprocessInputImage(image);
        if(tensorImage !=null) {
            trainingSamples.add(new TrainingSample(tensorImage, encoding(className)));
        }

    }

    public Iterator<List<TrainingSample>> trainingBatches(int batchSize) {
        return new Iterator<List<TrainingSample>>() {

            private int nextIndex = 0;

            @Override
            public boolean hasNext() {
                return nextIndex < trainingSamples.size();
            }

            @Override
            public List<TrainingSample> next() {
                int fromIndex = nextIndex;
                int toIndex = nextIndex + batchSize;
                nextIndex = toIndex;

                if (toIndex >= trainingSamples.size()){
                    return trainingSamples.subList(trainingSamples.size() - batchSize, trainingSamples.size());
                }else {
                    return trainingSamples.subList(fromIndex, toIndex);
                }
            }
        };
    }

    public TensorImage preprocessInputImage(Bitmap image) {
        int height = image.getHeight();
        int width = image.getWidth();
        int cropSize = Math.min(height, width);
        ImageProcessor imageProcessor= new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0.237f, 0.231f))
                .build();
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(image);
        TensorImage processedImage = imageProcessor.process(tensorImage);
        return processedImage;
    }

    public int encoding(String name) {
        int code = (name =="NO") ? 0 : 1;
        return code;
    }
}
