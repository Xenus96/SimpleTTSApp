# SimpleTTSApp version 2.0

<img width="1038" height="665" alt="Screenshot 2026-02-07 164316" src="https://github.com/user-attachments/assets/46b47d7c-5092-4c2c-900d-ab613ac8e4cc" />

## This is a simple XTTSv2-powered Text-to-Speech App written in Python, which has the following features: </br>
1. Make a voiceover of a block of text written in one of the following languages: **English, Russian, German, Romanian, French and Spanish**.
2. Allows to **add more text blocks** to render several sentences written in different languages. In this case, the resulting audio file will have all the rendered text blocks **concatenated**.
3. Each block of text use the **"Auto-Detect"** language feature by default if the user don't define the specific one.
4. Allows to set the **Maximum RAM Usage**, which tries to limit the SimpleTTSApp RAM consumption during the rendering process.
5. Allows to **import text** from a .txt file with the **"Import Text File"** button.
6. Allows to select the **Speaker** (or the voice type) which will be used to render the resulting audio.
7. Allows to **stop the rendering process** with the **"Stop Render"** button.
8. Allows to **save** the last rendered audio file via the **"Save Last Audio"** button.
9. Displays the total amount of time which was taken by the rendering process.

**IMPORTANT NOTE**: This is a draft version of the TTS App, which can have bugs and performance issues! In addition to that, the **"Max RAM Usage"** feature **MAY NOT GUARANTEE** that the set RAM limit is taken into account during the rendering process.

### Required tools:</br>
1. Microsoft C++ Build Tools **(latest)**;
2. Coqui XTTSv2 ML model;
3. Python 3.11 **(not newer)**;
4. the list of other required libraries is shown inside the **"import"** section of the "app.py" file.
