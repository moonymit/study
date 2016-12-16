package study.game2d.object;

import android.graphics.Bitmap;

/**
 * Created by Hi.JiGOO on 16. 12. 16..
 */
public abstract class GameObject {

    protected Bitmap IMAGE;

    protected final int WIDTH;
    protected final int HEIGHT;

    protected final int ROW_COUNT;
    protected final int COL_COUNT;

    protected int x;
    protected int y;

    protected final int width;
    protected final int height;

    public GameObject(Bitmap image, int rowCount, int colCount, int x, int y)  {

        this.IMAGE = image;

        this.ROW_COUNT= rowCount;
        this.COL_COUNT= colCount;

        this.WIDTH = image.getWidth();
        this.HEIGHT = image.getHeight();

        this.x= x;
        this.y= y;

        this.width = this.WIDTH/ colCount;
        this.height= this.HEIGHT/ rowCount;
    }

    protected Bitmap createSubImageAt(int row, int col)  {
        Bitmap subImage = Bitmap.createBitmap(IMAGE, col * width, row * height, width, height);
        return subImage;
    }

    public int getX()  {
        return this.x;
    }

    public int getY()  {
        return this.y;
    }


    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
}
