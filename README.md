# Распознавание типовых документов с помощью мультимодальных моделей

Исследовать:
- договоры
- товарные накладные
- счета на оплату
- паспорт
- таблицы с данными на изображении

Вытаскивать:
- реквизиты сторон
- список товаров
- суммы платежей
- срок действия договора

Формат на вход:
- pdf
- картинка
- промпт

Формат на выход:
- словарь с ключевой информацией из документа и мета-инфой 
(ббокс текста, возможно уверенность в распознавании текста)

## Tools
### Vision-Language Models (VLLMs)				
These models can understand and process both visual and textual data, useful for tasks like document layout understanding.				
- Donut -	Designed for structured document understanding, works well with documents like invoices, receipts, and contracts - NAVER - https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa
- Molmo -	A family of open vision-language models - https://huggingface.co/allenai/Molmo-7B-D-0924
- NVLM - Open Frontier-Class Multimodal LLMs - https://nvlm-project.github.io/
- Pix2Struct - Vision-language model, focused on document structure extraction, good for forms and charts - https://github.com/google-research/pix2struct
- Qwen2-VL - SoTA understanding of images of various resolution & ratio	Allen Institute for AI - https://github.com/QwenLM/Qwen2-VL
- XLM-Roberta -	Multilingual VLLM with fine-tuning potential for diverse text and document layouts - https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k
- LayoutLMv3 - Specialized in document layout understanding, with capabilities to recognize and classify text based on visual layout	Microsoft - https://huggingface.co/docs/transformers/model_doc/layoutlmv3
- -----------------------------
- LLaVA - Модель, которая объединяет языковые и визуальные способности для выполнения задач, связанных с пониманием изображений и текстов одновременно - Microsoft - https://llava-vl.github.io/

### Multimodal Neural Networks				
These models combine multiple modalities (text, vision, audio) for richer understanding and recognition.				
- ALIGN - Can be used for image-text similarity and for zero-shot image classification - https://huggingface.co/docs/transformers/model_doc/align
- BEiT-3 - Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks - https://github.com/microsoft/unilm/tree/master/beit3
- BLIP-2 - A multimodal model tailored to image-to-text applications, supporting tasks like document captioning - Salesforce - https://huggingface.co/docs/transformers/model_doc/blip-2
- CLIP - A model trained on image-text pairs that can be applied to document image retrieval and classification - OpenAI - https://github.com/openai/CLIP
- CoCa - Модель может фокусироваться на различных аспектах мультимодальных данных, улучшая точность аннотаций и описаний - https://github.com/lucidrains/CoCa-pytorch
- DocFormer - Применяется для понимания сложных документов и широко используется для задач, таких как классификация и информация - https://github.com/shabie/docformer - https://arxiv.org/abs/2106.11539
- Flamingo - Multimodal model for tasks that require simultaneous understanding of images and text - https://github.com/mlfoundations/open_flamingo - https://arxiv.org/abs/2204.14198
- FormNet - Performance on natural language and document understanding tasks - https://arxiv.org/abs/2203.08411
- LayoutLM - Обрабатывает текстовую информацию, визуальную информацию (размещение и форматирование текста), а также структуру документа, что позволяет извлекать данные из сложных многостраничных документов - Microsoft - https://huggingface.co/docs/transformers/model_doc/layoutlm
- MM1 - Способна решать задачи, связанные с изображениями и текстом, например, подсчет объектов или выполнение математических операций, используя методы рассуждения - https://arxiv.org/abs/2403.09611
- OFA - A multimodal framework useful for various document-based visual and text tasks, like image-text retrieval -https://github.com/OFA-Sys/OFA
- OmniFusion - Первая в России мультимодальная модель, способная поддерживать визуальный диалог и отвечать на вопросы по изображениям - AIRI - https://github.com/AIRI-Institute/OmniFusion
- PaLI - Генерация текстов на основе изображений и визуальный вопрос-ответ - https://github.com/kyegomez/PALI - https://arxiv.org/abs/2209.06794
- PandaGPT - Combines visual and language capabilities, enabling it to handle document tasks that require image-text interactions - Tencent AI Lab - https://panda-gpt.github.io/
- SimVLM - Используется для генерации описаний изображений и выполнения других задач мультимодального понимания - https://github.com/YulongBonjour/SimVLM - https://arxiv.org/abs/2108.10904
- StrucTexT - Structured text understanding on Visually Rich Documents (VRDs) is a crucial part of Document Intelligence - 	https://arxiv.org/abs/2108.02923
- TAPAS - For answering questions about tabular data - https://huggingface.co/docs/transformers/model_doc/tapas
- -----------------------------
- TILT - Arctic-TILT is a Snowflake-grown LLM that leverages a proprietary and unique transformer architecture, tailored to understand and extract data from documents - Snowflake - https://www.snowflake.com/en/blog/arctic-tilt-compact-llm-advanced-document-ai/
- UDOP - For document AI tasks like document image classification, document parsing and document visual question answering - https://huggingface.co/docs/transformers/model_doc/udop

### OCR Models for Document Recognition				
These OCR engines are suitable for extracting text from structured and unstructured documents, including contracts, invoices, and receipts.				
- DocTR - Robust two-stage OCR predictors that efficiently localize and recognize text in documents - https://github.com/mindee/doctr	
- EasyOCR - Built on PyTorch, it’s good for both handwritten and printed text across 80+ languages - https://github.com/JaidedAI/EasyOCR	
- Keras-OCR - An OCR pipeline using Keras and TensorFlow, suitable for custom document OCR workflows - https://keras-ocr.readthedocs.io/en/latest/	
- M4C - Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA - https://mmf.sh/docs/projects/m4c/	
- PaddleOCR - Advanced OCR engine with multilingual support and high accuracy, great for document digitization tasks - PaddlePaddle (Baidu) - https://github.com/PaddlePaddle/PaddleOCR	
- Tesseract - OCR	A widely used, reliable OCR tool that supports multiple languages, ideal for printed text recognition - https://github.com/tesseract-ocr/tesseract	
- TrOCR - Transformer-based OCR model that can handle handwritten as well as printed text - Microsoft - https://huggingface.co/docs/transformers/model_doc/trocr	
- SelfDoc - Использует подход самосупервизируемого обучения для обработки документов, извлекая информацию о структуре и содержании документа - https://arxiv.org/abs/2106.03331
- -----------------------------
- Adobe OCR - OCR for PDFs, making it ideal for document workflows where PDFs need to be converted to editable or searchable formats - Adobe - https://experienceleague.adobe.com/en/docs/document-cloud-learn/acrobat-learning/getting-started/scan-and-ocr	
- Amazon - Textract	A cloud-based OCR service that goes beyond simple text extraction, capable of identifying structured data such as tables, forms, and key-value pairs - Amazon - https://aws.amazon.com/textract/	
- CLaMM - Provides specialized OCR for historical and handwritten documents, useful for custom archival projects - https://clamm.irht.cnrs.fr/	
- Google Cloud Vision OCR - Part of its cloud-based image recognition services, supporting both printed and handwritten text in multiple languages - Google - https://cloud.google.com/vision/docs/ocr	
- Microsoft Azure OCR - Part of its cloud services, providing accurate text recognition in printed and handwritten text - Microsoft - https://learn.microsoft.com/ru-ru/azure/ai-services/computer-vision/overview-ocr	

- ### Large Language Models (LLMs)				
These are general-purpose language models that can be fine-tuned for document understanding tasks				
- Bloom -	A multilingual LLM, many languages - BigScience -	https://huggingface.co/bigscience/bloom
- Falcon -	LLM for summarization, question answering, and more	Technology Innovation Institute -	https://huggingface.co/tiiuae/falcon-mamba-7b
- Flan-T5 -	Instruction-tuned model based on T5, capable of zero-shot and fine-tuned document classification -	Google -	https://huggingface.co/google/flan-t5-base
- GPT-Neo and GPT-J -	versions inspired by GPT-3, useful for NLP tasks -	EleutherAI -	https://sapling.ai/llm/gpt-j-vs-gptneo
- Kosmos-1 -	A rudimentary reimplementation of the KOSMOS-1 model described in Microsofts paper -	https://github.com/bjoernpl/KOSMOS_reimplementation
- LLaMA -	Good for custom NLP tasks -	Meta -	https://github.com/meta-llama/llama3
- Mistral -	Suitable for document processing applications -	Mistral AI -	https://docs.mistral.ai/
- -----------------------------
- Claude -	Optimized for safe and interpretable responses -	Anthropic -	https://docs.anthropic.com/en/home
- Gemini -	Audio, images, videos, and text -	Google DeepMind -	https://ai.google.dev/gemini-api/docs
- GPT-4 -	Generative Pre-trained Transformer -	OpenAI -	https://platform.openai.com/docs/introduction
- Grok -	A variety of tasks, including generating and understanding text, code, and function calling -	xAI -	https://x.ai/api				
				
### ICDAR 2024 Proceedings	
The 18th International Conference on Document Analysis and Recognition (ICDAR) features numerous papers on cutting-edge topics such as document image processing, layout analysis, and text recognition - https://icdar2024.net/procceedings/


## Companies

- Smart Engines
https://smartengines.ru/clients/
https://smartengines.ru/blog/
    * Скорость распознавания – 15 страниц в сек без GPU (4.6-битные нейросети)
    * Нейросетевая OCR на основе квазисимвологий
    * Геометрически осведомленный ИИ (Geometry-Aware AI)
    * Распознавание печатных и рукописных документов на 102 языках мира за счет использования уникальных моделей синтеза обучающих данных
    * Распознавание сканов и фотографий, автоматическая классификация и сортировка документов
    * Распознавание печатных и рукописных реквизитов, таблиц, чекбоксов, штрихкодов
    * On-premise – поставляется в виде SDK с API для интеграции в различные системы (ERP, RPA, ECM, CRM, АБС и т.д.) и мобильные приложения. Без риска утечки данных и коммерческой тайны
    * Аутентификация бланков, проверка действительности документов, выявления цифровых подделок. Проверка цвета и наличия подписей и печатей

- Beorg Smart Vision
https://beorg.ru/kadrovye-buhgalterskie-documenty/
https://beorg.ru/blog/
    * Распознавание паспортов (качество 99+%, распознавание рукописных данных)
    * Распознавание комплектов персональных документов (ИНН, СНИЛС, заявления, трудовые книжки, водительские удостоверения и пр.)
    * Первичная бухгалтерская документация  и др.
Ключевым моментом при этом является верификация оцифрованных данных силами операторов краудсорсинговой платформы.

- Directum
https://www.directum.ru/products/directum/intelligence
    * Распознавание + человеческая верификация
https://ecm-journal.ru/material/pochemu_servis_100-nogo_raspoznavanija_ehto_vishenka_na_torte_bukhgalterskogo_dokumentooborota
- Энтера
https://entera.pro/
- 1С-Софт
https://portal.1c.ru/applications/1C-Document-Recognition
https://v8.1c.ru/its/services/1s-raspoznavanie-pervichnykh-dokumentov/
- Content AI
https://contentai.ru/


## Статьи
Для чего нужно распознавание в банковской сфере?
https://smartengines.ru/blog/dlya-chego-nuzhno-raspoznavanie-v-bankovskoy-sfere/

Публикационная активность и интеллектуальная деятельность Smart Engines за первое полугодие 2024 год
https://smartengines.ru/science/publications/#web_of_science

Как мы поставили точку в распознавании паспорта, посадив программистов за прописи
https://smartengines.ru/blog/kak-my-postavili-tochku-v-raspoznavanii-pasporta-posadiv-programmistov-za-propisi/

Что такое разметка данных и для чего она нужна?
https://beorg.ru/blog/chto-takoe-razmetka-dannyh-i-dlya-chego-ona-nuzhna/

Распознавание паспортов и других документов: OCR на практике
https://beorg.ru/blog/raspoznavanie-dokumentov-ocr-na-praktike/

Поиск четырёхугольников документов на мобильных устройствах
https://habr.com/ru/companies/smartengines/articles/260533/

Биполярные морфологические сети: нейрон без умножения
https://habr.com/ru/companies/smartengines/articles/497310/










