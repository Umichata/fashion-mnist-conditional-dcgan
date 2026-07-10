<a id="ru"></a>

# Сравнение условных DCGAN на Fashion MNIST

**Русский** | [English](#en)

Проект посвящен исследованию условной генерации изображений одежды из датасета Fashion MNIST с помощью двух архитектур DCGAN.

Обе модели получают случайный вектор скрытого пространства и метку класса, после чего генерируют изображение одного из 10 классов Fashion MNIST. Исследование сравнивает компактный базовый генератор и более сложный усиленный генератор при одинаковом условном проекционном дискриминаторе, одинаковых функциях потерь и едином цикле обучения.

Проект оформлен как воспроизводимый notebook pipeline: разведочный анализ данных, адаптивная конфигурация Google Colab для GPU T4, L4 и A100, обучение независимого оценочного классификатора, построение двух GAN, проверка градиентов, обучение с EMA-весами, количественная оценка, визуальное сравнение, интерполяция в скрытом пространстве и проверка ближайших соседей.

## Ноутбуки

- [Русская версия ноутбука](notebooks/dcgan_fashion_mnist_research_ru.ipynb)
- [English notebook version](notebooks/dcgan_fashion_mnist_research_en.ipynb)

## Цель исследования

Цель проекта - определить, дает ли более сложный генератор практическое преимущество над компактной условной DCGAN на изображениях Fashion MNIST размером `28 x 28`.

Сравнение проводится в одинаковых условиях:

- один датасет и одинаковая предобработка;
- один условный проекционный дискриминатор;
- одинаковая hinge-функция потерь;
- одинаковые правила формирования меток;
- одинаковая стратегия аугментации;
- одинаковые обратные вызовы;
- одинаковая система метрик;
- оценка лучших EMA-весов генераторов.

## Датасет

В проекте используется Fashion MNIST из `tf.keras.datasets`.

Датасет содержит:

- 60 000 обучающих изображений;
- 10 000 тестовых изображений;
- изображения размером `28 x 28`;
- один канал яркости;
- 10 классов одежды и обуви.

Классы:

1. футболка или топ;
2. брюки;
3. пуловер;
4. платье;
5. пальто;
6. сандалия;
7. рубашка;
8. кроссовок;
9. сумка;
10. ботинок.

## Сравниваемые модели

### Базовая условная DCGAN

Компактный генератор использует:

- вектор скрытого пространства;
- вложение метки класса;
- полносвязное преобразование в карту признаков `7 x 7`;
- транспонированные свертки;
- пакетную нормализацию;
- `LeakyReLU`;
- выходной слой с `tanh`.

### Усиленная условная DCGAN

Усиленный генератор использует:

- более широкую начальную карту признаков;
- вложение метки класса большей размерности;
- увеличение пространственного разрешения;
- обычные свертки после масштабирования;
- дополнительные блоки уточнения признаков;
- пакетную нормализацию;
- `ReLU`;
- выходной слой с `tanh`.

### Общий дискриминатор

Обе модели используют один и тот же условный проекционный дискриминатор. Он оценивает:

- реалистичность изображения;
- соответствие изображения заданной метке класса.

Такой дизайн позволяет сравнивать именно архитектуры генераторов, не смешивая результат с различиями в дискриминаторах или тренировочной логике.

## Методика обучения

В проекте используются:

- hinge-функция потерь;
- оптимизатор Adam;
- смешанная точность на поддерживаемых GPU;
- экспоненциальное скользящее среднее весов генератора;
- легкий шум на входе дискриминатора;
- пространственная аугментация;
- ограничение нормы градиентов;
- снижение скорости обучения при плато;
- ранняя остановка;
- сохранение лучшего EMA-генератора;
- проверка существования и конечности градиентов;
- автоматическая остановка при численной нестабильности.

Перед длительным обучением выполняется проверка одного шага, которая подтверждает:

- наличие градиентов;
- изменение весов генератора;
- изменение весов дискриминатора;
- конечность функций потерь;
- корректность пользовательского цикла обучения.

## Оценка качества

Оценочный классификатор обучается отдельно и не участвует в оптимизации генераторов. Он используется только для независимой оценки результатов.

В проекте рассчитываются:

- FID-подобная оценка в признаковом пространстве классификатора;
- итоговая агрегированная оценка качества;
- энтропия покрытия классов;
- соответствие заданным меткам;
- минимальная доля класса;
- число пропущенных классов;
- разнообразие признаков;
- отношение полной вариации к реальным данным;
- отношение сильных пиксельных перепадов к реальным данным;
- показатели устойчивости градиентов;
- время выполнения эпохи;
- расстояния до ближайших реальных изображений.

## Результаты

Репрезентативные запуски на NVIDIA L4 показали:

| Метрика | Базовая условная DCGAN | Усиленная условная DCGAN |
|---|---:|---:|
| FID-подобная оценка, меньше лучше | `0.16-0.18` | `0.44-0.47` |
| Итоговая оценка качества, больше лучше | `0.984-0.985` | `0.971-0.972` |
| Соответствие заданным меткам | около `0.97-0.98` | около `0.98` |
| Энтропия покрытия классов | около `1.0` | около `1.0` |
| Пропущенные классы | `0` | `0` |
| Среднее время эпохи на L4 | около `18-19` секунд | около `31` секунды |

Результаты могут незначительно меняться между запусками из-за стохастической природы GAN.

Основной вывод исследования: базовая условная DCGAN показала лучший баланс качества, устойчивости и вычислительной эффективности. Усиленная архитектура корректно обучилась и обеспечила полное покрытие классов, но ее дополнительная сложность не дала преимущества на изображениях Fashion MNIST размером `28 x 28`.

## Структура проекта

```text
fashion-mnist-conditional-dcgan/
├── README.md
├── LICENSE
├── .gitignore
└── notebooks/
    ├── dcgan_fashion_mnist_research_ru.ipynb
    └── dcgan_fashion_mnist_research_en.ipynb
```

После выполнения ноутбук сохраняет экспериментальные артефакты в каталог:

```text
/content/dcgan_fashion_mnist_final/
├── config.json
├── images/
├── models/
└── tables/
```

Сохраняются:

- конфигурация эксперимента;
- веса генераторов;
- EMA-веса генераторов;
- веса дискриминаторов;
- история обучения;
- итоговые метрики;
- метрики по классам;
- таблица устойчивости;
- изображения для визуального анализа.

## Запуск

1. Откройте русскую или английскую версию ноутбука в Google Colab.
2. Выберите аппаратный ускоритель GPU.
3. Выполните все ячейки последовательно.
4. Дождитесь обучения оценочного классификатора и двух GAN-моделей.
5. Изучите итоговые таблицы, графики и сгенерированные изображения.
6. Скачайте сохраненные артефакты из `/content/dcgan_fashion_mnist_final`.

Ноутбук автоматически подбирает профиль для:

- NVIDIA T4;
- NVIDIA L4;
- NVIDIA A100;
- CPU для сокращенного проверочного запуска.

## Основные зависимости

- Python 3;
- TensorFlow / Keras;
- NumPy;
- pandas;
- Matplotlib;
- SciPy;
- Google Colab с GPU рекомендуется для полного запуска.

## Воспроизводимость

В проекте фиксируются случайные начальные состояния Python, NumPy и TensorFlow. При этом полное совпадение численных результатов между разными GPU, версиями CUDA и версиями TensorFlow не гарантируется, поскольку обучение GAN остается стохастическим.

---

<a id="en"></a>

# Comparison of Conditional DCGANs on Fashion MNIST

[Русский](#ru) | **English**

This project studies conditional fashion-image generation on the Fashion MNIST dataset using two DCGAN architectures.

Both models receive a latent-space random vector and a class label, then generate an image belonging to one of the 10 Fashion MNIST classes. The study compares a compact baseline generator with a deeper enhanced generator while keeping the conditional projection discriminator, loss functions, and training loop identical.

The project is implemented as a reproducible notebook pipeline: exploratory data analysis, adaptive Google Colab configuration for T4, L4, and A100 GPUs, training an independent evaluation classifier, building two GANs, validating gradients, EMA-based training, quantitative evaluation, visual comparison, latent-space interpolation, and nearest-neighbor analysis.

## Notebooks

- [Russian notebook version](notebooks/dcgan_fashion_mnist_research_ru.ipynb)
- [English notebook version](notebooks/dcgan_fashion_mnist_research_en.ipynb)

## Research Objective

The objective is to determine whether a more complex generator provides a practical advantage over a compact conditional DCGAN on `28 x 28` Fashion MNIST images.

The comparison uses identical experimental conditions:

- the same dataset and preprocessing;
- the same conditional projection discriminator;
- the same hinge loss;
- the same label-generation rules;
- the same augmentation strategy;
- the same callbacks;
- the same evaluation system;
- evaluation of the best EMA generator weights.

## Dataset

The project uses Fashion MNIST from `tf.keras.datasets`.

The dataset contains:

- 60,000 training images;
- 10,000 test images;
- `28 x 28` images;
- one grayscale channel;
- 10 clothing and footwear classes.

Classes:

1. T-shirt or top;
2. trouser;
3. pullover;
4. dress;
5. coat;
6. sandal;
7. shirt;
8. sneaker;
9. bag;
10. ankle boot.

## Compared Models

### Baseline Conditional DCGAN

The compact generator uses:

- a latent-space vector;
- a class-label embedding;
- a dense projection to a `7 x 7` feature map;
- transposed convolutions;
- batch normalization;
- `LeakyReLU`;
- a `tanh` output layer.

### Enhanced Conditional DCGAN

The enhanced generator uses:

- a wider initial feature map;
- a larger class-label embedding;
- spatial upsampling;
- standard convolutions after upsampling;
- additional feature-refinement blocks;
- batch normalization;
- `ReLU`;
- a `tanh` output layer.

### Shared Discriminator

Both models use the same conditional projection discriminator. It evaluates:

- image realism;
- consistency between the image and the requested class label.

This design isolates the generator architecture as the primary experimental variable.

## Training Method

The project uses:

- hinge loss;
- the Adam optimizer;
- mixed precision on supported GPUs;
- exponential moving averages of generator weights;
- light discriminator input noise;
- spatial augmentation;
- gradient-norm clipping;
- learning-rate reduction on plateau;
- early stopping;
- best EMA-generator checkpointing;
- gradient presence and finiteness diagnostics;
- automatic termination on numerical instability.

Before long training begins, a one-step validation confirms:

- gradients are present;
- generator weights change;
- discriminator weights change;
- losses are finite;
- the custom training loop operates correctly.

## Quality Evaluation

The evaluation classifier is trained independently and does not participate in generator optimization. It is used only to evaluate generated samples.

The project reports:

- an FID-like score in the classifier feature space;
- an aggregate quality score;
- class-coverage entropy;
- requested-label agreement;
- minimum class share;
- missing-class count;
- feature diversity;
- total-variation ratio relative to real data;
- high-contrast transition ratio relative to real data;
- gradient-stability indicators;
- epoch duration;
- nearest-real-image distances.

## Results

Representative runs on an NVIDIA L4 produced:

| Metric | Baseline Conditional DCGAN | Enhanced Conditional DCGAN |
|---|---:|---:|
| FID-like score, lower is better | `0.16-0.18` | `0.44-0.47` |
| Aggregate quality score, higher is better | `0.984-0.985` | `0.971-0.972` |
| Requested-label agreement | approximately `0.97-0.98` | approximately `0.98` |
| Class-coverage entropy | approximately `1.0` | approximately `1.0` |
| Missing classes | `0` | `0` |
| Mean epoch time on L4 | approximately `18-19` seconds | approximately `31` seconds |

Results may vary slightly between runs because GAN training is stochastic.

The main research conclusion is that the baseline conditional DCGAN provides the best balance of generation quality, numerical stability, and computational efficiency. The enhanced architecture trains correctly and covers all classes, but its additional complexity does not improve results on `28 x 28` Fashion MNIST images.

## Project Structure

```text
fashion-mnist-conditional-dcgan/
├── README.md
├── LICENSE
├── .gitignore
└── notebooks/
    ├── dcgan_fashion_mnist_research_ru.ipynb
    └── dcgan_fashion_mnist_research_en.ipynb
```

After execution, the notebook stores experimental artifacts in:

```text
/content/dcgan_fashion_mnist_final/
├── config.json
├── images/
├── models/
└── tables/
```

Saved artifacts include:

- experiment configuration;
- generator weights;
- EMA generator weights;
- discriminator weights;
- training histories;
- final metrics;
- per-class metrics;
- stability diagnostics;
- visual-analysis images.

## Running the Project

1. Open the Russian or English notebook in Google Colab.
2. Select a GPU hardware accelerator.
3. Run all cells in order.
4. Wait for the evaluation classifier and both GAN models to finish training.
5. Review the final tables, charts, and generated images.
6. Download the saved artifacts from `/content/dcgan_fashion_mnist_final`.

The notebook automatically selects a hardware profile for:

- NVIDIA T4;
- NVIDIA L4;
- NVIDIA A100;
- CPU for a shortened validation run.

## Main Dependencies

- Python 3;
- TensorFlow / Keras;
- NumPy;
- pandas;
- Matplotlib;
- SciPy;
- Google Colab with a GPU is recommended for a full run.

## Reproducibility

The project fixes random seeds for Python, NumPy, and TensorFlow. Exact numerical equality across different GPUs, CUDA versions, and TensorFlow versions is not guaranteed because GAN training remains stochastic.
