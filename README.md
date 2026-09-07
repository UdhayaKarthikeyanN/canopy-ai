<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Issues][issues-shield]][issues-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/UdhayaKarthikeyanN/canopy-ai">
    <img src="docs/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Canopy AI</h3>

  <p align="center">
    Fully local AI-powered urban tree canopy and heat analysis
    <br />
    <a href="#how-it-works"><strong>Explore how it works »</strong></a>
    <br />
    <br />
    <a href="#screenshots">View Screenshots</a>
    &middot;
    <a href="https://github.com/UdhayaKarthikeyanN/canopy-ai/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/UdhayaKarthikeyanN/canopy-ai/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#configurable-parameters">Configurable Parameters</a></li>
    <li><a href="#training-the-model">Training The Model</a></li>
    <li><a href="#tests">Tests</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Overview dashboard][product-screenshot]](#screenshots)

Canopy AI takes a satellite or aerial image (JPG/PNG) and works out, pixel by pixel,
where the tree canopy already is, where new trees could realistically go, and how much
cooling that would buy you. Upload an image and it performs pixel-level land-cover
segmentation, detects plantable areas, generates a recommended planting plan with
realistic tree spacing, computes heat metrics with before/after projections, and
exports a professional PDF report — all in one pass, in a few seconds, on your CPU.

**No VLMs, no LLMs, no external AI APIs, no cloud services, no map APIs.** Per-pixel
classification is a small locally-trained Random Forest (scikit-learn, ~11 MB, trained
entirely on synthetic labeled scenes — see [Training the model](#training-the-model)),
followed by classical computer-vision post-processing. Results are deterministic and
reproducible: same image and settings in, same recommendation out, every time.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.org]][Python-url]
* [![Flask][Flask.com]][Flask-url]
* [![OpenCV][OpenCV.org]][OpenCV-url]
* [![scikit-learn][scikit-learn.org]][scikit-learn-url]
* [![NumPy][NumPy.org]][NumPy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SCREENSHOTS -->
## Screenshots

| | |
|---|---|
| ![Landing page](docs/screenshots/landing.png) | ![Overview dashboard](docs/screenshots/overview.png) |
| ![Analysis maps](docs/screenshots/analysis-maps.png) | ![Before/after comparison](docs/screenshots/before-after.png) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Canopy AI runs entirely on your own machine — no account, no API key, no internet
connection required once dependencies are installed.

### Prerequisites

* Python 3.10+
* Windows, macOS, or Linux (the one-click launcher below is Windows-only; everything
  else works cross-platform)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/UdhayaKarthikeyanN/canopy-ai.git
   cd canopy-ai
   ```
2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Launch it
   ```sh
   python app.py
   ```
   On Windows you can instead just double-click **`run_canopy.bat`** — it creates an
   isolated virtual environment, installs dependencies on first run, and opens your
   browser automatically. Use **`stop_canopy.bat`** to stop it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Open **http://127.0.0.1:7860** in your browser, drop in a satellite or aerial JPG/PNG,
adjust the calibration parameters if needed, and hit **Analyze Image**. The dashboard
gives you:

* **Analysis Maps** — the original image, AI land-cover segmentation, detected
  plantable area, and the recommended planting plan, each full-screen previewable.
* **Overview** — canopy/plantable/heat-score KPIs, land-cover composition, and an
  environmental impact summary (CO₂ sequestration, new shade, impervious surface).
* **Before / After** — heat score, temperature, and canopy cover, projected against
  the recommended planting scenario.
* A one-click **PDF report** covering all of the above, generated locally.

The server binds to `127.0.0.1` only — nothing ever leaves your machine.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- HOW IT WORKS -->
## How It Works

1. **Segmentation** (`engine/segmentation.py`, `engine/features.py`) — an
   edge-preserving bilateral filter stabilises spectra; every pixel is scored with
   vegetation indices (ExG, VARI), HSV hue/saturation/value, RGB and an 11x11 local
   texture statistic. A locally-trained Random Forest classifier
   (`engine/model/land_cover_rf.joblib`) assigns trees, grass, buildings, roads/paved,
   water and bare/open soil from those 9 features. SLIC superpixels then aggregate
   labels into perceptually uniform regions; a majority (mode) filter removes
   salt-and-pepper noise; connected-component analysis deletes regions below 0.02% of
   the scene; shadow spectra are re-assigned by hue/saturation-weighted neighbourhood
   voting; compact-vs-elongated shape analysis separates buildings from road segments.
   All masks keep the original image resolution and align 1:1 with input pixels.
2. **Plantable mask** (`engine/planting.py`) — grass + bare-soil pixels with an exact
   Euclidean distance-transform buffer of 1.5 m around trees, buildings and roads
   (2.5 m around water). Fragments below 0.04% of the scene are discarded as noise.
3. **Tree placement** — deterministic Bridson Poisson-disk sampling inside the eroded
   plantable mask enforces strict minimum spacing, so markers never overlap existing
   trees, buildings or roads.
4. **Metrics** — percentages come from actual pixel counts at the configured ground
   resolution (m/pixel). Heat score = clamp(18 + 0.90·impervious% + 0.40·bare% −
   0.70·canopy% − 0.20·grass%, 0–100). Temperature = baseline − 0.06°C per canopy %
   point (published urban-forestry range 0.04–0.10).
5. **PDF report** (`engine/report.py`) — original image, AI segmentation,
   plantable-area image, planting plan, all metrics, before/after heat comparison,
   methodology and environmental impact (CO₂, shade).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONFIGURABLE PARAMETERS -->
## Configurable Parameters

Available in the Advanced section of the UI:

| Parameter | Default | Meaning |
|---|---|---|
| Baseline temperature | 33°C | Regional baseline used by the heat model |
| Ground resolution | 0.5 m/px | Ground sample distance of the image |
| Tree spacing | 7 m | Minimum distance between planted trees |
| Mature crown radius | 3.5 m | Crown size used for projected canopy |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TRAINING THE MODEL -->
## Training The Model

`engine/model/land_cover_rf.joblib` is committed to the repo, so a normal clone needs
no training step. To retrain it (e.g. after changing `engine/features.py` or the
synthetic generator):

```sh
python engine/train_model.py
```

This generates synthetic scenes on the fly (`tests/make_test_image.py`, each with
randomized hue/saturation/brightness jitter), using the exact per-pixel ground truth
the generator draws (every building/road/tree/water/soil primitive is mirrored onto a
label canvas as it's rendered — no manual labeling involved), trains a
`RandomForestClassifier`, reports held-out pixel accuracy on unseen jittered seeds, and
overwrites the `.joblib` file.

No real satellite imagery is used or required — this is a genuine limitation: the
classifier has only ever seen synthetic renders, so it generalizes less reliably to
real photographs than a model trained on real labeled imagery would.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TESTS -->
## Tests

```sh
python tests/make_test_image.py   # generate synthetic satellite image
python tests/test_engine.py       # engine validation (masks, spacing, metrics, PDF)
python tests/test_api.py          # HTTP API end-to-end
```

Each analysis is stored under `outputs/<analysis_id>/`: `original.png`,
`segmentation.png`, `plantable.png`, `recommendation.png`, `report.pdf`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Local pixel-level land-cover segmentation
- [x] Plantable-area detection with obstacle buffers
- [x] Deterministic Poisson-disk tree placement
- [x] Heat score / temperature before-after projections
- [x] PDF report export
- [ ] Training data from real labeled imagery, not just synthetic scenes
- [ ] Placement strategy that prioritizes maximum cooling impact over even coverage
- [ ] Batch analysis across multiple images / a whole district

See the [open issues](https://github.com/UdhayaKarthikeyanN/canopy-ai/issues) for a
full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create
a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Udhaya Karthikeyan — [github.com/UdhayaKarthikeyanN](https://github.com/UdhayaKarthikeyanN)

Project Link: [https://github.com/UdhayaKarthikeyanN/canopy-ai](https://github.com/UdhayaKarthikeyanN/canopy-ai)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) — this
  README's structure is adapted from it
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [OpenCV](https://opencv.org), [scikit-learn](https://scikit-learn.org),
  [ReportLab](https://www.reportlab.com/opensource/) — the local CV/ML/PDF stack this
  project is built on

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[issues-shield]: https://img.shields.io/github/issues/UdhayaKarthikeyanN/canopy-ai.svg?style=for-the-badge
[issues-url]: https://github.com/UdhayaKarthikeyanN/canopy-ai/issues
[stars-shield]: https://img.shields.io/github/stars/UdhayaKarthikeyanN/canopy-ai.svg?style=for-the-badge
[stars-url]: https://github.com/UdhayaKarthikeyanN/canopy-ai/stargazers
[license-shield]: https://img.shields.io/github/license/UdhayaKarthikeyanN/canopy-ai.svg?style=for-the-badge
[license-url]: https://github.com/UdhayaKarthikeyanN/canopy-ai/blob/master/LICENSE
[product-screenshot]: docs/screenshots/overview.png
[Python.org]: https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Flask.com]: https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[OpenCV.org]: https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[scikit-learn.org]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/
[NumPy.org]: https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
