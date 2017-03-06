package com.edgedetector.pedrock.aplicativomobilecanny;

import android.os.*;
import android.speech.tts.TextToSpeech;

import java.util.Locale;

/**
 * Created by Pedrock on 06/11/2016.
 */

public class Tutorial extends MainActivity {

    private TextToSpeech speechObjetivos, speechFeedbackCurto, speechFeedbackLongo;

    private Tutorial() {
    }

    protected void tutorial(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final android.os.Vibrator vibrator = (android.os.Vibrator) getSystemService(this.VIBRATOR_SERVICE);



        //Tutorial de uso e informações doo sistema passadas por voz sintetica
        speechObjetivos = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR) {
                    speechObjetivos.setLanguage(Locale.US);

                }
                String objetivo = "\n" + "Obrigado por usar o sistema," + "\n" + "Para usufluir melhor verifique as seguintes coisas."
                        + "\n" + "Primeiro, tenha certeza de está segurando o celular na posição Vertical..."
                        + "\n" + "Segundo, verifique com as mãos se nada está obstruindo a lente da câmera traseira..."
                        + "\n" + "Agora, com a parte traseira voltada para frente, segure o celular proximo ao peito, bom uso..."
                        + "\n" + "Esse sistema temo como objetivo passar uma noção de distancia identificando com a câmera possiveis obstaculos no caminho," + "\n"
                        + "\n" + "Para isso faz uso da vibração do celular.";
                speechObjetivos.speak(objetivo, TextToSpeech.QUEUE_FLUSH, null);
            }
        });

        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override
            public void run() {
                speechFeedbackCurto = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
                    @Override
                    public void onInit(int status) {
                        if (status != TextToSpeech.ERROR) {
                            speechFeedbackCurto.setLanguage(Locale.US);

                        }
                        String objetivo = "\n" + "Para indicar possiveis obstaculos proximos, o celular emite uma vibração curta, assim";
                        speechFeedbackCurto.speak(objetivo, TextToSpeech.QUEUE_FLUSH, null);
                        while (speechFeedbackCurto.isSpeaking()) {
                            vibrator.cancel();
                        }
                        vibrator.vibrate(300);
                    }
                });
            }
        }, 1000);

        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override
            public void run() {
                speechFeedbackLongo = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
                    @Override
                    public void onInit(int status) {
                        if (status != TextToSpeech.ERROR) {
                            speechFeedbackLongo.setLanguage(Locale.US);

                        }
                        String objetivo = "\n" + "Para indicar possiveis obstaculos longes, o celular emite uma vibração longa, assim";
                        speechFeedbackLongo.speak(objetivo, TextToSpeech.QUEUE_FLUSH, null);
                        while (speechFeedbackLongo.isSpeaking()) {
                            vibrator.cancel();
                        }
                        vibrator.vibrate(600);
                    }
                });
            }
        }, 40000);
    }
}