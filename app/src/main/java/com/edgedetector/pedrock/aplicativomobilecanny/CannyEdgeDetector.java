package com.edgedetector.pedrock.aplicativomobilecanny;

import android.graphics.Bitmap;

import java.util.Arrays;

public class CannyEdgeDetector extends  MainActivity{

// variaveis estaticas

    private final static float GAUSSIAN_CUT_OFF = 0.005f;
    private final static float MAGNITUDE_SCALE = 10F;
    private final static float MAGNITUDE_LIMIT = 100F;
    private final static int MAGNITUDE_MAX = (int) (MAGNITUDE_SCALE * MAGNITUDE_LIMIT);

// variaveis

    private int height;
    private int width;
    private int picsize;
    private int[] data;
    private int[] magnitude;
    private Bitmap sourceImage;
    private Bitmap edgesImage;

    private float gaussianKernelRadius;
    private float lowThreshold;
    private float highThreshold;
    private int gaussianKernelWidth;
    private boolean contrastNormalized;

    private float[] xConv;
    private float[] yConv;
    private float[] xGradient;
    private float[] yGradient;

// construtores

    /**
     * Cronstroi o Detector com esses parametros.
     */

    public CannyEdgeDetector() {
        lowThreshold = 2.5f;
        highThreshold = 10f;
        gaussianKernelRadius = 2f;
        gaussianKernelWidth = 16;
        contrastNormalized = false;
    }

// Acessorios

    /**
     * A imagem que fornece os dados de luminância usados ​​por este detector de bordas
     *
     * @return a imagem, ou null
     */

    public Bitmap getSourceImage() {
        return sourceImage;
    }

    /**
     *Especifica a imagem que fornecerá os dados de luminância em que as bordas serão detectadas.
     *Uma imagem de origem deve ser definida antes do método de processo ser chamado.
     *
     * @param image a origem dos dados de luminância
     */

    public void setSourceImage(Bitmap image) {
        // Converte para RGB
        sourceImage = image.copy(Bitmap.Config.ARGB_8888, true);
    }

    /**
     * Obtém uma imagem contendo as bordas detectadas durante a última chamada do método de process.
     * A imagem armazenada em buffer é uma imagem opaca do tipo
     * BufferedImage.TYPE_INT_ARGB em que os pixels de borda são brancos e todos os outros pixels são pretos.
     *
     * @return uma imagem com bordas detectadas, ou null se o método process não foi chamada.
     */

    public Bitmap getEdgesImage() {
        return edgesImage;
    }

    /**
     * Método usado apenas para referenciar na memória a imagem com bordas detectadas
     *
     * @param edgesImage imagem com bordas detectadas
     */

    public void setEdgesImage(Bitmap edgesImage) {
        this.edgesImage = edgesImage;
    }

    /**
     * O low threshold para histerese. O valor é 2.5.
     *
     * @return O low histerese limitador
     */

    public float getLowThreshold() {
        return lowThreshold;
    }

    /**
     * Sets o low threshold para histerese.
     *
     * @param threshold  O low histerese limitador
     */

    public void setLowThreshold(float threshold) {
        if (threshold < 0) throw new IllegalArgumentException();
        lowThreshold = threshold;
    }

    /**
     * O high threshold  para histerese. O valor é 10.
     *
     * @return O high histerese limitador
     */

    public float getHighThreshold() {
        return highThreshold;
    }

    /**
     * Sets o high threshold  para histerese.
     *
     * @param threshold O high histerese limitador
     */

    public void setHighThreshold(float threshold) {
        if (threshold < 0) throw new IllegalArgumentException();
        highThreshold = threshold;
    }

    /**
     * Número de pixel que é aplicado o filtro gaussiano.
     *
     * @return O raio da operação de convolução em pixels
     */

    public int getGaussianKernelWidth() {
        return gaussianKernelWidth;
    }

    /**
     * O número de pixels em que o filtro Gaussiano é aplicado.
     * Esta implementação irá reduzir o raio se a contribuição de valores de pixel da imagem for considerada negligenciável,
     * então este é realmente um raio máximo.
     *
     * @param gaussianKernelWidth Um raio para a operação de convolução em pixels, pelo menos 2.
     */

    public void setGaussianKernelWidth(int gaussianKernelWidth) {
        if (gaussianKernelWidth < 2) throw new IllegalArgumentException();
        this.gaussianKernelWidth = gaussianKernelWidth;
    }

    /**
     * O raio do filtro gaussiano de convolução usado para suavizar a imagem fonte antes do cálculo do gradiente.
     * O valor é 16.
     *
     * @return o raio do filtro gaussiano
     */

    public float getGaussianKernelRadius() {
        return gaussianKernelRadius;
    }

    /**
     * Sets O raio do filtro gaussiano de convolução usado para suavizar a imagem fonte antes do cálculo do gradiente..
     *
     * @return um raio do filtro gaussiano, que exceda 0.1f.
     */

    public void setGaussianKernelRadius(float gaussianKernelRadius) {
        if (gaussianKernelRadius < 0.1f) throw new IllegalArgumentException();
        this.gaussianKernelRadius = gaussianKernelRadius;
    }


// métodos

    public void process() {
        width = sourceImage.getWidth(); //Tamanho é pego da imagem de origem para se adaptar ao celular
        height = sourceImage.getHeight(); //Tamanho é pego da imagem de origem para se adaptar ao celular
        picsize = width * height; //Tamanho total da imagem de origem
        initArrays();
        readLuminance();
        computeGradients(gaussianKernelRadius, gaussianKernelWidth);
        int low = Math.round(lowThreshold * MAGNITUDE_SCALE);
        int high = Math.round(highThreshold * MAGNITUDE_SCALE);
        performHysteresis(low, high);
        thresholdEdges();
        writeEdges(data);
    }

// métodos privados

    private void initArrays() { //Inicialização das variaveis necessarias para calculo
        if (data == null || picsize != data.length) {
            data = new int[picsize];
            magnitude = new int[picsize];

            xConv = new float[picsize];
            yConv = new float[picsize];
            xGradient = new float[picsize];
            yGradient = new float[picsize];
        }
    }

    private void computeGradients(float kernelRadius, int kernelWidth) {

        //geração das mascaras
        float kernel[] = new float[kernelWidth];
        float diffKernel[] = new float[kernelWidth];
        int kwidth;
        for (kwidth = 0; kwidth < kernelWidth; kwidth++) {
            float g1 = gaussian(kwidth, kernelRadius);
            if (g1 <= GAUSSIAN_CUT_OFF && kwidth >= 2) break;
            float g2 = gaussian(kwidth - 0.5f, kernelRadius);
            float g3 = gaussian(kwidth + 0.5f, kernelRadius);
            kernel[kwidth] = (g1 + g2 + g3) / 3f / (2f * (float) Math.PI * kernelRadius * kernelRadius);
            diffKernel[kwidth] = g3 - g2;
        }

        int initX = kwidth - 1;
        int maxX = width - (kwidth - 1);
        int initY = width * (kwidth - 1);
        int maxY = width * (height - (kwidth - 1));

        //faz a covolução nas direções X e Y da imagem
        for (int x = initX; x < maxX; x++) {
            for (int y = initY; y < maxY; y += width) {
                int index = x + y;
                float sumX = data[index] * kernel[0];
                float sumY = sumX;
                int xOffset = 1;
                int yOffset = width;
                for (; xOffset < kwidth; ) {
                    sumY += kernel[xOffset] * (data[index - yOffset] + data[index + yOffset]);
                    sumX += kernel[xOffset] * (data[index - xOffset] + data[index + xOffset]);
                    yOffset += width;
                    xOffset++;
                }

                yConv[index] = sumY;
                xConv[index] = sumX;
            }

        }

        for (int x = initX; x < maxX; x++) {
            for (int y = initY; y < maxY; y += width) {
                float sum = 0f;
                int index = x + y;
                for (int i = 1; i < kwidth; i++)
                    sum += diffKernel[i] * (yConv[index - i] - yConv[index + i]);

                xGradient[index] = sum;
            }

        }

        for (int x = kwidth; x < width - kwidth; x++) {
            for (int y = initY; y < maxY; y += width) {
                float sum = 0.0f;
                int index = x + y;
                int yOffset = width;
                for (int i = 1; i < kwidth; i++) {
                    sum += diffKernel[i] * (xConv[index - yOffset] - xConv[index + yOffset]);
                    yOffset += width;
                }

                yGradient[index] = sum;
            }

        }

        initX = kwidth;
        maxX = width - kwidth;
        initY = width * kwidth;
        maxY = width * (height - kwidth);
        for (int x = initX; x < maxX; x++) {
            for (int y = initY; y < maxY; y += width) {
                int index = x + y;
                int indexN = index - width;
                int indexS = index + width;
                int indexW = index - 1;
                int indexE = index + 1;
                int indexNW = indexN - 1;
                int indexNE = indexN + 1;
                int indexSW = indexS - 1;
                int indexSE = indexS + 1;

                float xGrad = xGradient[index];
                float yGrad = yGradient[index];
                float gradMag = hypot(xGrad, yGrad);

                //Execução da supressão não-máxima
                float nMag = hypot(xGradient[indexN], yGradient[indexN]);
                float sMag = hypot(xGradient[indexS], yGradient[indexS]);
                float wMag = hypot(xGradient[indexW], yGradient[indexW]);
                float eMag = hypot(xGradient[indexE], yGradient[indexE]);
                float neMag = hypot(xGradient[indexNE], yGradient[indexNE]);
                float seMag = hypot(xGradient[indexSE], yGradient[indexSE]);
                float swMag = hypot(xGradient[indexSW], yGradient[indexSW]);
                float nwMag = hypot(xGradient[indexNW], yGradient[indexNW]);
                float tmp;

                if (xGrad * yGrad <= (float) 0 /*(1)*/
                        ? Math.abs(xGrad) >= Math.abs(yGrad) /*(2)*/
                        ? (tmp = Math.abs(xGrad * gradMag)) >= Math.abs(yGrad * neMag - (xGrad + yGrad) * eMag) /*(3)*/
                        && tmp > Math.abs(yGrad * swMag - (xGrad + yGrad) * wMag) /*(4)*/
                        : (tmp = Math.abs(yGrad * gradMag)) >= Math.abs(xGrad * neMag - (yGrad + xGrad) * nMag) /*(3)*/
                        && tmp > Math.abs(xGrad * swMag - (yGrad + xGrad) * sMag) /*(4)*/
                        : Math.abs(xGrad) >= Math.abs(yGrad) /*(2)*/
                        ? (tmp = Math.abs(xGrad * gradMag)) >= Math.abs(yGrad * seMag + (xGrad - yGrad) * eMag) /*(3)*/
                        && tmp > Math.abs(yGrad * nwMag + (xGrad - yGrad) * wMag) /*(4)*/
                        : (tmp = Math.abs(yGrad * gradMag)) >= Math.abs(xGrad * seMag + (yGrad - xGrad) * sMag) /*(3)*/
                        && tmp > Math.abs(xGrad * nwMag + (yGrad - xGrad) * nMag) /*(4)*/
                        ) {
                    magnitude[index] = gradMag >= MAGNITUDE_LIMIT ? MAGNITUDE_MAX : (int) (MAGNITUDE_SCALE * gradMag);
                } else {
                    magnitude[index] = 0;
                }
            }
        }
    }

    private float hypot(float x, float y) {
        return (float) Math.hypot(x, y);
    }

    private float gaussian(float x, float sigma) {
        return (float) Math.exp(-(x * x) / (2f * sigma * sigma));
    }

    private void performHysteresis(int low, int high) {
        Arrays.fill(data, 0);

        int offset = 0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (data[offset] == 0 && magnitude[offset] >= high) {
                    follow(x, y, offset, low);
                }
                offset++;
            }
        }
    }

    private void follow(int x1, int y1, int i1, int threshold) {
        int x0 = x1 == 0 ? x1 : x1 - 1;
        int x2 = x1 == width - 1 ? x1 : x1 + 1;
        int y0 = y1 == 0 ? y1 : y1 - 1;
        int y2 = y1 == height - 1 ? y1 : y1 + 1;

        data[i1] = magnitude[i1];
        try {
            for (int x = x0; x <= x2; x++) {
                for (int y = y0; y <= y2; y++) {
                    int i2 = x + y * width;
                    if ((y != y1 || x != x1)
                            && data[i2] == 0
                            && magnitude[i2] >= threshold) {
                        follow(x, y, i2, threshold);
                    }
                }
            }
        } catch (StackOverflowError e) {
            e.printStackTrace();
        }
        return;
    }

    private void thresholdEdges() {
        for (int i = 0; i < picsize; i++) {
            data[i] = data[i] > 0 ? -1 : 0xff000000;
        }
    }

    private int luminance(float r, float g, float b) {
        return Math.round(0.299f * r + 0.587f * g + 0.114f * b);
    }

    private void readLuminance() {
        if (true) {
            int[] pixels = new int[picsize];
            sourceImage.getPixels(pixels, 0, width, 0, 0, width, height);
            for (int i = 0; i < picsize; i++) {
                int p = pixels[i];
                int r = (p & 0xff0000) >> 16;
                int g = (p & 0xff00) >> 8;
                int b = p & 0xff;
                data[i] = luminance(r, g, b);
            }
        }
    }

    private void normalizeContrast() {
        int[] histogram = new int[256];
        for (int i = 0; i < data.length; i++) {
            histogram[data[i]]++;
        }
        int[] remap = new int[256];
        int sum = 0;
        int j = 0;
        for (int i = 0; i < histogram.length; i++) {
            sum += histogram[i];
            int target = sum * 255 / picsize;
            for (int k = j + 1; k <= target; k++) {
                remap[k] = i;
            }
            j = target;
        }

        for (int i = 0; i < data.length; i++) {
            data[i] = remap[data[i]];
        }
    }

    private void writeEdges(int pixels[]) {
        if (edgesImage == null) {
            edgesImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        }
        edgesImage.setPixels(pixels, 0, width, 0, 0, width, height);
    }
}

