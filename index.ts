import * as faceapi from 'face-api.js';
import { Canvas, Image, ImageData, loadImage } from 'canvas';
import * as fs from 'fs';
import * as path from 'path';

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function run(): Promise<void> {
    const imagePath = process.argv[2];
    if (!imagePath) {
        console.error('Image path is required.');
        process.exit(1);
    }
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

                ctx.strokeStyle = 'lime';
                ctx.lineWidth = 4;
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

        // 出力ディレクトリの存在をチェックし、なければ作成
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }
        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(outputPath, buffer);
        console.log(`Saved to ${outputPath}`);
    } catch (error) {
        console.error(error);
        process.exit(1);
    }
}

run();