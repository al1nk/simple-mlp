package perceptron;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import mnisttools.MnistReader;

public class ImageConverter {

	private static final float DEFAULT = 100;


	public static float[] image2VecteurBinaire(int[][] image, float seuil){
		float[] x  = new float[image.length*image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				if (image[i][j] > seuil){
					x[k] = 1;
				}
				else
					x[k] = 0;
			}
		}
		return x;
	}
	public static float[] image2VecteurReel(int[][] image){
		float[] x  = new float[image.length*image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				x[k]=image[i][j];
			}
		}
		return x;
	}
	
	public static float[] image2VecteurReel_withB(int[][] image){
		float[] x  = new float[image.length*image[0].length+1];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j+1;
				x[k]=image[i][j];
			}
		}
		x[0] = 1;
		return x;
	}
	
	
	public static int[] float2IntVector(float []x){
		int[] y = new int[x.length];
		for (int i = 0; i < y.length; i++) {
			y[i] = (int) x[i];
		}
		return y;
	}
	
	public static int[] image2IntVector(int[][] image){
		int[] x  = new int[image.length*image[0].length];

		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				int k = image.length*i+j;
				x[k] = image[i][j];
			}
		}
		return x;
	}
	
	public static float[] image2VecteurBinaire(int[][] image){
		return image2VecteurBinaire(image, DEFAULT);
	}


	public static void main(String[] args) {
		String path = System.getProperty("user.dir")+"/";
		String labelDB=path+"train-labels-idx1-ubyte";
		String imageDB=path+"train-images-idx3-ubyte";
		MnistReader db = new MnistReader(labelDB, imageDB);
		int idx = 300; // une variable pour stocker l'index
		// Attention la premiere valeur est 1. 
		int [][] image = db.getImage(idx);
		float[] x = ImageConverter.image2VecteurReel(image);
		int l = 15;
		int sizeL = image[l].length;
		for (int i = 0; i < sizeL ; i++) {
			//System.out.print(x[i+l*sizeL]+", ");
			assert(x[i+l*sizeL] == image[l][i]);
		}
		System.exit(0);
		System.out.println("ici");
		System.out.println(Arrays.toString(image[l]));
		int numberOfColumns = 28;//image.length;
		int numberOfRows = 28; //image[0].length;
        int[] vimage = new int[numberOfColumns*numberOfRows]; //image2intVector(image);
        for (int i = 0; i < numberOfColumns; i++) {
            for (int j = 0; j < numberOfRows; j++) {
                float pxl = x[i*numberOfColumns + j]/255;
                vimage[i*numberOfColumns + j] = new Color(pxl, pxl, pxl).getRGB();
            }
        }
        BufferedImage bimage = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
        bimage.setRGB(0, 0, numberOfColumns, numberOfRows, vimage, 0, numberOfColumns);

		File outputfile = new File(path+"imagesChiffresMannuscrits.png");
		System.out.println("l�");
		try {
			System.out.println("l�");
			ImageIO.write(bimage, "png", outputfile);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

