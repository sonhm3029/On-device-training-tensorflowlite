package com.example.tf_lite_ex;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.ClipData;
import android.content.Intent;
import android.database.Cursor;
import android.icu.text.SimpleDateFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.provider.Settings;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.protobuf.ByteString;

import org.tensorflow.lite.examples.model.ModelController;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.examples.transport.ClientRequest;
import io.grpc.examples.transport.Parameters;
import io.grpc.examples.transport.ServerReply;
import io.grpc.examples.transport.TransportGrpc;
import io.grpc.stub.StreamObserver;

public class MainActivity extends AppCompatActivity {

    private static final String Tag = "TF_LITE_LOG";
    private ModelController modelController;
    private File imageFolder;
    private File yesFolder;
    private File noFolder;
    private static final int REQUEST_IMAGE_PICKER_YES = 10;
    private static final int REQUEST_IMAGE_PICKER_NO = 20;
    private static final int MANAGE_EXTERNAL_STORAGE_PERMISSION_REQUEST_CODE = 101;
    private ManagedChannel channel;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        Log.i(Tag, "OnCreate");
        modelController = new ModelController(MainActivity.this);
        setContentView(R.layout.firstactivity);
        resultText = (TextView) findViewById(R.id.grpc_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());

        imageFolder = new File(getExternalFilesDir(null), "brain");

        if(!imageFolder.exists()) {
            Log.i(Tag, "CREATE NEW FOLDER!");
            imageFolder.mkdirs();
            yesFolder = new File(imageFolder, "YES");
            noFolder = new File(imageFolder, "NO");
            yesFolder.mkdirs();
            noFolder.mkdirs();
        }
        yesFolder = new File(imageFolder, "YES");
        noFolder = new File(imageFolder, "NO");


        Button trainingButton = findViewById(R.id.button);
        trainingButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                connect(view);
            }
        });

        Button btnImportYes = findViewById(R.id.browseYesData);
        btnImportYes.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openImagePicker("YES");
            }
        });

        Button btnImportNo = findViewById(R.id.browseNoData);
        btnImportNo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openImagePicker("NO");
            }
        });

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Log.d(Tag, "REQUEST PERMISION");
            requestManageExternalStoragePermission();
        } else {
            // Proceed with your regular flow for devices below Android 11
            // For older versions, you can access external storage without any special permission
        }

        Button btnTest = findViewById(R.id.button2);
        btnTest.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runGrpc(v);
            }
        });

    }

    @RequiresApi(api = Build.VERSION_CODES.R)
    private void requestManageExternalStoragePermission() {
        if (!Environment.isExternalStorageManager()) {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Permission Required");
            builder.setMessage("This app requires access to manage external storage. Please grant the permission in the settings.");
            builder.setPositiveButton("Open Settings", (dialog, which) -> {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                Uri uri = Uri.fromParts("package", getPackageName(), null);
                intent.setData(uri);
                startActivityForResult(intent, MANAGE_EXTERNAL_STORAGE_PERMISSION_REQUEST_CODE);
            });
            builder.setNegativeButton("Cancel", (dialog, which) -> {
                // Handle if the user cancels the permission request
            });
            builder.show();
        } else {
            // The permission is already granted
        }
    }


    private void openImagePicker(String type) {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, type=="YES"? REQUEST_IMAGE_PICKER_YES : REQUEST_IMAGE_PICKER_NO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_PICKER_YES && resultCode == RESULT_OK) {
            if (data != null && data.getData() != null) {
                Uri selectedImageUri = data.getData();
                saveImageToAppStorage(selectedImageUri, yesFolder);
            }
        }
        if (requestCode == REQUEST_IMAGE_PICKER_NO && resultCode == RESULT_OK) {
            if (data != null && data.getData() != null) {
                Uri selectedImageUri = data.getData();
                saveImageToAppStorage(selectedImageUri, noFolder);
            }
        }

        if (requestCode == MANAGE_EXTERNAL_STORAGE_PERMISSION_REQUEST_CODE) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    // The permission has been granted, you can now access the external storage
                } else {
                    // The user did not grant the permission
                    // Handle the case when the permission is not granted
                }
            }
        }

    }

    private void saveImageToAppStorage(Uri imageUri, File currentSaveFolder) {
        try {
            // Open the InputStream to read the selected image
            InputStream inputStream = getContentResolver().openInputStream(imageUri);

            // Create a new file in the app's storage folder
            Log.d(Tag, currentSaveFolder.getAbsolutePath());
            File imageFile = new File(currentSaveFolder, "image_" + System.currentTimeMillis() + ".jpg");

            // Open a FileOutputStream to write the image data to the file
            FileOutputStream outputStream = new FileOutputStream(imageFile);

            // Copy the image data from InputStream to FileOutputStream
            int imageSize = inputStream.available();
            byte[] buffer = new byte[imageSize];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            Toast.makeText(MainActivity.this,"Successfully make import image", Toast.LENGTH_LONG).show();
            // Close the streams
            inputStream.close();
            outputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }



    public static String getCurrentTime() {
        Calendar calendar = Calendar.getInstance();
        int hour = calendar.get(Calendar.HOUR_OF_DAY);
        int minute = calendar.get(Calendar.MINUTE);
        int second = calendar.get(Calendar.SECOND);

        // Format the time components manually
        String formattedTime = String.format("%02d:%02d:%02d", hour, minute, second);
        return formattedTime;
    }
    public void setResultText(String text) {
        String time = getCurrentTime();
        resultText.append("\n" + time + "   " + text);
    }


    //            10.0.52.65
    //            192.168.1.7
    public void connect(View view) {
        channel = ManagedChannelBuilder.forAddress("10.0.52.143", 50051).maxInboundMessageSize(10 * 1024 * 1024).usePlaintext().build();
        setResultText("Channel object created. Ready to train!");
    }

    public void runGrpc(View view) {
        MainActivity activity = this;
        ExecutorService executor = Executors.newSingleThreadExecutor();

        Handler handler = new Handler(Looper.getMainLooper());

        executor.execute(new Runnable() {
            private String result;
            @Override
            public void run() {
                try {
                    (new IvirseServiceRunnale()).run(TransportGrpc.newStub(channel), activity);
                }catch (Exception e) {
                    StringWriter sw = new StringWriter();
                    PrintWriter pw = new PrintWriter(sw);
                    e.printStackTrace(pw);
                    pw.flush();
                    result = "Failed to connect to the FL server \n" + sw;
                }
                handler.post(() -> {
                    setResultText(result);
                });
            }
        });
    }

    private static class IvirseServiceRunnale{
        protected Throwable failed;
        private StreamObserver<ClientRequest> requestObserver;

        public  void run(TransportGrpc.TransportStub asyncStub, MainActivity activity) {
            join(asyncStub, activity);
        }

        private void join(TransportGrpc.TransportStub asyncStub, MainActivity activity)
                throws RuntimeException{

            final CountDownLatch finishLatch = new CountDownLatch(1);
            requestObserver = asyncStub.join(new StreamObserver<ServerReply>() {
                @Override
                public void onNext(ServerReply value) {
                    handleMessage(value, activity);
                }

                @Override
                public void onError(Throwable t) {
                    t.printStackTrace();
                    failed = t;
                    finishLatch.countDown();
                    Log.e(Tag, t.getMessage());
                }

                @Override
                public void onCompleted() {
                    finishLatch.countDown();
                    Log.e(Tag, "Done");
                }
            });
        }

        private void handleMessage(ServerReply message, MainActivity activity) {
            try {
                ClientRequest request = ClientRequest.newBuilder().build();
                requestObserver.onNext(request);
            }catch (Exception e) {
                Log.e(Tag, e.getMessage());
            }
        }
    }
}