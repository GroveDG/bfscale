# bfscale

Best Fit Scale (bfscale) is a image downscaling method to create the highest possible image quality upon upscaling/upsampling with a certain scaling method (at the moment only bilinear). This is achieved by curve fitting the sections of the original image that correspond to pixels in the downscaled image.

Bfscale was originally developed for improved Signed Distance Fields (SDFs), but was separated due to its broader usefulness.

<center>
  <img alt="Billiard Balls A. Unscaled" src="https://github.com/GroveDG/bfscale/assets/87248833/83cc6055-9e07-441e-9972-e800e7b7bf81" width=30% height=30%>
  <img alt="Billiard Balls A. Bfscaled" src="https://github.com/GroveDG/bfscale/assets/87248833/03dd0a20-09c5-4a2b-8bc1-d58448ea7707" width=30% height=30%>
  <img alt="Billiard Balls A. Linear Scaled" src="https://github.com/GroveDG/bfscale/assets/87248833/8ca32b3d-cfcb-4243-b8d2-d76e9838b08c" width=30% height=30%>


  <img alt="Billiard Balls A. Unscaled Cropped" src="https://github.com/GroveDG/bfscale/assets/87248833/a1906415-febf-4fcc-ac89-71ce5aa67423)" width=30% height=30%>
  <img alt="Billiard Balls A. Bfscaled Cropped" src="https://github.com/GroveDG/bfscale/assets/87248833/3fb0ebb5-c9c4-4cbd-803d-44484c1af2f6)" width=30% height=30%>
  <img alt="Billiard Balls A. Linear Cropped" src="https://github.com/GroveDG/bfscale/assets/87248833/c973ed69-e30e-4267-b207-60bb2f6c1e20)" width=30% height=30%>
</center>


The images shown are part of the TESTIMAGES project: http://testimages.tecnick.com
Copyright (C) 2011-2014 Nicola Asuni - Tecnick.com LTD
