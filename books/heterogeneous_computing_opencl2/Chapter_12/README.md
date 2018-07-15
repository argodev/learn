# Chapter 12 - WebCL: Enabling OpenCL Acceleration of Web Applications

This chapter provides a high-level overview of WebCL - a Web-focused version of OpenCL. This runtime is limited in nature and designed to support calculations - not graphics. Graphics support using GPUs and similar are provided in WebGL (a completely separate project).

While this is conceptually quite interesting to me, and I have a project wherein it might have utility (client-side calculation and rendering of the location and coverage pattern of satellite constellations), it appears that the browser support for such is quite low.

The text went on to discuss server-side support in platforms such as Node.js and even showed step-by-step instructions for setting it up and running it. Unfortunately, other than maybe the ease of the extant javascript bindings, I don't see the benefit of using WebCL on the server when I could likely just use OpenCL directly which is more full-featured.