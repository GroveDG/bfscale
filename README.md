# bfscale

Best Fit Scale (bfscale) is a image downscaling method to create the highest possible image quality upon upscaling/upsampling with a certain scaling method (in this instance bilinear). This is achieved by fitting the parameters of the upscaling model to sections of the original image. In the case of bilinear, this can be done with a simple linear regression.

Bfscale was originally developed for improved Signed Distance Fields (SDFs), but was separated due to its broader usefulness.


## Dependencies
<ul>
  <li>Numpy</li>
  <li>Numba</li>
  <li>ImageIO</li>
  <li>PySimpleGUI</li>
</ul>

## Image Comparison

<table>
  <tr>
    <th>Unscaled</th>
    <th>Bfscale 20% Scale</th>
    <th>Linear 20% Scale</th>
  </tr>
  <tr>
    <td><img alt="Billiard Balls A. Unscaled" src="https://github.com/GroveDG/bfscale/assets/87248833/c7cd6e43-19ee-42fd-b5c6-9bda1a5ca9cf" width="10000"></td>
    <td><img alt="Billiard Balls A. Bfscaled" src="https://github.com/GroveDG/bfscale/assets/87248833/2d009578-bb3e-41f2-9a61-929046fa8b24" width="10000"></td>
    <td><img alt="Billiard Balls A. Linear Scaled" src="https://github.com/GroveDG/bfscale/assets/87248833/d3e676e0-0d85-48b9-bbfa-8424deb805b1" width="10000"></td>
  </tr>
  <tr>
    <td><img alt="Billiard Balls A. Unscaled Cropped" src="https://github.com/GroveDG/bfscale/assets/87248833/694e895e-371f-46f2-b402-e73b217b4e3e" width="10000"></td>
    <td><img alt="Billiard Balls A. Bfscaled Cropped" src="https://github.com/GroveDG/bfscale/assets/87248833/4c58f538-9e88-4f3a-a81c-8d1a4ab37437" width="10000"></td>
    <td><img alt="Billiard Balls A. Linear Cropped" src="https://github.com/GroveDG/bfscale/assets/87248833/85bd749a-caa5-43a2-bd15-056542cfe61c" width="10000"></td>
  </tr>
</table>
<details>
  <summary>Image Source</summary>
  <p>Copyright (C) 2011-2014 Nicola Asuni - Tecnick.com LTD</p>
  <p>The images shown are part of the TESTIMAGES project: http://testimages.tecnick.com</p>
</details>
