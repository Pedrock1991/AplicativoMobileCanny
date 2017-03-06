package com.edgedetector.pedrock.aplicativomobilecanny;

import android.graphics.Bitmap;

import org.opencv.core.Core;

/**
 * Created by Pedrock on 06/11/2016.
 */

public class Vibrator extends MainActivity {

    final android.os.Vibrator vibrator = (android.os.Vibrator) getSystemService(this.VIBRATOR_SERVICE);

    private Vibrator() {
    }

    public void obstacleCanny(Bitmap Image, int Width, int Height){
        int color;
        int Max = Width*Height;
        int divisão = Height/3;
        int[] pixels = new int[Max];
        boolean Colision = false;
        Image.getPixels(pixels, 0, Width, 0, 0, Width, Height);

        while(true){
            for(int i=2*divisão*Width;i<Max;i++){ //parte inferior da imagem
                if(pixels[i]==0xff){
                    if(pixels[i+1]==0xff || pixels[i*Width]==0xff || pixels[i+1*Width]==0xff){
                        vibrator.vibrate(300);
                        Colision = true;
                        break;
                    }
                }
            }
            if(Colision == false) {
                for (int i = divisão * Width; i < 2 * divisão * Width; i++) { //parte meio da imagem
                    if (pixels[i] == 0xff) {
                        if (pixels[i + 1] == 0xff || pixels[i * Width] == 0xff || pixels[i + 1 * Width] == 0xff) {
                            vibrator.vibrate(600);
                            break;
                        }
                    }
                }
            }
        }

    }

    public void obstacleSobel(Bitmap Image, int Width, int Height){
        int color;
        int Max = Width*Height;
        int[] pixels = new int[Max];
        int divisão = Height/3;
        int densidade=0;
        int j;
        boolean Colision = false;
        Image.getPixels(pixels, 0, Width, 0, 0, Width, Height);

        while(true){
            for(int i=2*divisão*Width;i<Max;i++){ //parte inferior da imagem
                if(pixels[i]==0xff){
                    densidade+=1;
                    for(j=i;j<i+10;j++) {
                        if (pixels[j + 1] == 0xff || pixels[j * Width] == 0xff || pixels[j + 1 * Width] == 0xff) {
                            densidade += 1;
                        }
                    }
                }
                if(densidade>20) {
                    vibrator.vibrate(300);
                    Colision = true;
                    break;
                }else densidade=0;
            }

            if(Colision == false) {
                for (int i = divisão * Width; i < 2 * divisão * Width; i++) { //parte meio da imagem
                    if (pixels[i] == 0xff) {
                        densidade += 1;
                        for(j=i;j<i+10;j++) {
                            if (pixels[j + 1] == 0xff || pixels[j * Width] == 0xff || pixels[j + 1 * Width] == 0xff) {
                                densidade += 1;
                            }
                        }
                    }
                    if (densidade > 20) {
                        vibrator.vibrate(600);
                        break;
                    }else densidade=0;
                }
            }
        }

    }
}
