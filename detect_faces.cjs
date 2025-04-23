const faceapi = require('face-api.js');
const { Canvas, Image, ImageData, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function run() {
    const imagePath = process.argv[2];
    const outputDir = 'output';
    const outputFilename = `output_${path.basename(imagePath)}`;
    const outputPath = path.join(outputDir, outputFilename);
    try {
        const modelsPath = path.resolve(__dirname, 'models');
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
        await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath);
        const absoluteImagePath = path.resolve(imagePath);
        const img = await loadImage(absoluteImagePath);

        // Canvas を作成して画像を描画
        const canvas = new Canvas(img.width, img.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, img.width, img.height);
        const detections = await faceapi.detectAllFaces(canvas).withFaceLandmarks();

        if (detections.length > 0) {
            console.log(`${detections.length} faces were detected.`);

            // 検出された顔の輪郭を描画
            detections.forEach(detection => {
                const landmarks = detection.landmarks;
                const jawOutline = landmarks.getJawOutline();

                // 輪郭を描画
                ctx.strokeStyle = 'lime'; // 線の色 (緑)
                ctx.lineWidth = 4;       // 線の太さ
                ctx.beginPath();
                ctx.moveTo(jawOutline[0].x, jawOutline[0].y);
                for (let i = 1; i < jawOutline.length; i++) {
                    ctx.lineTo(jawOutline[i].x, jawOutline[i].y);
                }
                ctx.stroke();
            });

        } else {
            console.log('No face was detected in the image.');
        }

        // 結果をファイルに保存
        const buffer = canvas.toBuffer('image/png'); // PNG形式でバッファを取得
        fs.writeFileSync(outputPath, buffer);
        console.log(`save to ${outputPath}`);

    } catch (error) {
        console.error(error);
        process.exit(1); // エラーで終了
    }
}

run()