package org.tensorflow.lite.examples.model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.TensorFlowLite;
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
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.security.Signature;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.example.tf_lite_ex.R;
import com.google.protobuf.ByteString;

import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.examples.greet.GreeterGrpc;
import io.grpc.examples.greet.HelloReply;
import io.grpc.examples.greet.HelloRequest;
import io.grpc.examples.greet.Parameters;
import io.grpc.examples.greet.TestReply;
import io.grpc.examples.greet.TestRequest;
import io.grpc.stub.StreamObserver;

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
    private ByteBuffer modelParameters;
    private ByteBuffer nextModelParameters;
    private static final int FLOAT_BYTES = 4;

    private String GET_WEIGHT_SIG = "get_trainable_weights";
    private String UPDATE_WEIGHTS_SIG = "update_weights";

    // GRPC TASK
    private ManagedChannel channel;




    public ModelController(Context context){

        this.context = context;
        if( setupModel(context)) {
            targetHeight = interpreter.getInputTensor(0).shape()[2];
            targetWidth = interpreter.getInputTensor(0).shape()[1];

            int modelParameterSize = interpreter.getInputTensor(0).numElements();
            int bufferSize = modelParameterSize * FLOAT_BYTES;
            modelParameters = allocateBuffer(bufferSize);
            nextModelParameters = allocateBuffer(bufferSize);


            Log.i(TAG, "WIDTH " + targetWidth + "HEIght " + targetHeight);
        }
    }

    public boolean setupModel(Context context) {
        Log.i(TAG, "SETUP MODEL...");
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(numThreads);
            MappedByteBuffer modelFile = FileUtil.loadMappedFile(context, "model_v7.tflite");
            interpreter = new Interpreter(modelFile, options);
            Log.i(TAG, "Load model successfully!");
            return true;
        }catch (IOException e) {
            Log.e(TAG, "TFlite failed to load model with error: " + e.getMessage());
            return false;
        }
    }

    Throwable failed;
    public void testModel() {
        try {
            float[][] params = getWeights();

            float[] weights = params[0];

            ByteBuffer buffer = ByteBuffer.allocate(weights.length * 4);
            for (float value : weights) {
                buffer.putFloat(value);
            }

            buffer.rewind();

            List<ByteString> layers = new ArrayList<>();
            layers.add(ByteString.copyFrom(buffer));
            Parameters p = Parameters.newBuilder().addTensors(ByteString.copyFrom(buffer)).setTensorType("ND").build();
            HelloRequest request = HelloRequest.newBuilder().setParameters(p).build();


            channel = ManagedChannelBuilder.forAddress("192.168.1.7", 50051).usePlaintext().build();
            GreeterGrpc.GreeterStub stub = GreeterGrpc.newStub(channel);
            Log.i(TAG, "SUCESS CONNECT TO CHANNEL");

            StreamObserver<HelloRequest> requestObserver;
            final CountDownLatch finishLatch = new CountDownLatch(1);

            requestObserver = stub.interactingHello(
                    new StreamObserver<HelloReply>() {
                        @Override
                        public void onNext(HelloReply value) {
                            Parameters newPrams = value.getParameters();

                            Log.i(TAG, newPrams.getTensorType());
                        }

                        @Override
                        public void onError(Throwable t) {
                            t.printStackTrace();
                            failed = t;
                            finishLatch.countDown();
                            Log.i(TAG, "ON ERROR STreeam");
                            Log.e(TAG, t.getMessage());
                        }

                        @Override
                        public void onCompleted() {
                            finishLatch.countDown();
                            Log.i(TAG, "DONE GRPC");
                        }
                    }
            );

            // Send multiple messages
            for (int i = 1; i <= 5; i++) {
                requestObserver.onNext(request);
            }

            // Signal the end of requests
            requestObserver.onCompleted();

        }catch (Exception e) {
            Log.i(TAG, "ERROR" + e.getMessage());
        }

    }

    public HelloRequest weightsAsProto(ByteBuffer[] weights) {
        List<ByteString> layers = new ArrayList<>();
        for (ByteBuffer weight: weights) {
            layers.add(ByteString.copyFrom(weight));
        }
        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        return HelloRequest.newBuilder().setParameters(p).build();
    }

    public float[][] getWeights() {
        Map<String, Object> inputs = new HashMap<>();
        Map<String, Object> outputs = new HashMap<>();

        FloatBuffer weights = FloatBuffer.allocate(1280);
        FloatBuffer bias = FloatBuffer.allocate(1);

        inputs.put("str", "OK");
        outputs.put("weights",weights);
        outputs.put("bias", bias);
        interpreter.runSignature(inputs, outputs, GET_WEIGHT_SIG);

        float[] weightsArr = weights.array();
        float[] biasF = bias.array();

        return new float[][]{weightsArr, biasF};
    }

    public void updateWeights(float[][] params) {
        try {
            float[] weights = params[0];
            float[] bias = params[1];

            Map<String, Object> inputs = new HashMap<>();
            Map<String, Object> outputs = new HashMap<>();

            float[][] weights2D = new float[1280][1];
            for(int i =0; i< weights.length; i++) {
                weights2D[i][0] = weights[i];
            }


            inputs.put("w", weights2D);
            inputs.put("b", bias);

            IntBuffer out = IntBuffer.allocate(1);
            outputs.put("message", out);

            interpreter.runSignature(inputs, outputs, UPDATE_WEIGHTS_SIG);

            Log.i(TAG, "SUCCESS UPDATE WEIGHT");
        }catch (Exception e){
            Log.e(TAG, "ERROR UPDATE WEIGHT" + e.getMessage());
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

    public static float[] divideArray(float[] arr, int divisor) {
        float[] result = new float[arr.length];
        for (int i = 0; i< arr.length; i++) {
            result[i] = arr[i] / divisor;
        }

        return result;
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

    public ByteBuffer getParameters()  {
        return modelParameters;
    }

    public int encoding(String name) {
        int code = (name =="NO") ? 0 : 1;
        return code;
    }

    private static ByteBuffer allocateBuffer(int capacity) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        buffer.order(ByteOrder.nativeOrder());
        return buffer;
    }

}
