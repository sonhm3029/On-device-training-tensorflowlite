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
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.examples.model.ModelController;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private String Tag = "TF_LITE_LOG";
    private ModelController modelController;
    private File imageFolder;
    private File yesFolder;
    private File noFolder;
    private static final int REQUEST_IMAGE_PICKER_YES = 10;
    private static final int REQUEST_IMAGE_PICKER_NO = 20;
    private static final int MANAGE_EXTERNAL_STORAGE_PERMISSION_REQUEST_CODE = 101;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.i(Tag, "OnCreate");
        modelController = new ModelController(MainActivity.this);
        setContentView(R.layout.firstactivity);

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
                modelController.startTraining();
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


            // Now, you have the image saved in your app's storage folder (imageFile)
            // You can further process or display the image as needed
            // For example, you can create a Bitmap from the file and pass it to your classifyImage function
//            Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
//            classifyImage(bitmap);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        Log.i(Tag, "onStart");
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
        Log.i(Tag, "onResume");

    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.i(Tag, "onPause");

    }

    @Override
    protected void onRestart() {
        super.onRestart();
        Log.i(Tag, "onRestart");

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(Tag, "onDestroy");

    }
}