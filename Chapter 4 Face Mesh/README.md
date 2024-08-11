# mp.solutions.face_mesh

`mpFaceMesh = mp.solutions.face_mesh` imports the MediaPipe face mesh module for detecting and processing facial landmarks.

`faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)` initializes the face mesh model, set to detect up to two faces.

`results = faceMesh.process(imgRGB)` processes the image to detect face meshes.

Parameters for `the mpFaceMesh.FaceMesh()` class include: `self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon`.

- **staticMode**: Whether to treat each image as a static image. If True, face detection is performed on every frame. If False, it tracks the detected face, which is faster.
- **maxFaces**: Maximum number of faces to detect.
- **minDetectionCon**: Minimum confidence threshold for detection. Faces below this confidence are ignored.
- **minTrackCon**: Minimum confidence threshold for tracking. Tracking below this confidence is ignored.

# mp.solutions.drawing_utils

`mpDraw = mp.solutions.drawing_utils` imports the MediaPipe drawing utilities module.

`drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)` creates a drawing specification object to define the style for landmarks and connections, where thickness specifies line thickness and circle_radius specifies the radius of landmark points.

`mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)` calls MediaPipe's drawing function to draw landmarks and connections on the image.

- **img**: The image on which to draw landmarks.
- **faceLms**: The detected face landmarks collection.
- **mpFaceMesh.FACEMESH_TESSELATION**: Specifies the type of connections to draw; here, it's the face mesh tessellation.
- **drawSpec**: Defines the drawing style for landmarks and connections (e.g., thickness and circle radius). The first drawSpec parameter defines the style for landmarks (keypoints), such as circle radius and color, while the second defines the style for connections, such as line thickness and color.

# Issues in Source Code

These bugs won't cause errors during execution but will appear during debugging.

## 1、Parameter Name Change in New Version

Before:

![原来](./pics/原来.png)

Now:

![现在](./pics/现在.png)

## 2、Positional Argument Needs to Be a Keyword Argument

![关键字传参](./pics/关键字传参.png)