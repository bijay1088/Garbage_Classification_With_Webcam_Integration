function garbage_classifier_app()
    loadedData = load('trainedGarbageClassNet.mat');
    netTransfer = loadedData.netTransfer;
    
    cam = webcam;
    fig = uifigure('Name', 'Garbage Classifier', 'Position', [100 100 800 600]);
    img = uiimage(fig, 'Position', [50 100 700 400]);
    lbl = uilabel(fig, 'Position', [50 50 700 50], 'FontSize', 20);
    t = timer('ExecutionMode', 'fixedRate', 'Period', 0.1, 'TimerFcn', @(~,~) updateImage());

    start(t);
    
    function updateImage()
        frame = snapshot(cam);
        img.ImageSource = frame;
        [imds, ~] = preprocessImage(frame);
        label = classify(netTransfer, imds);
        lbl.Text = sprintf('Type: %s', string(label));
    end

    function [imds, imgResized] = preprocessImage(img)
        imgResized = imresize(img, [227 227]);
        imds = augmentedImageDatastore([227 227], imgResized);
    end

    function closeApp()
        stop(t);
        delete(t);
        clear cam;
        delete(fig);
    end

    fig.CloseRequestFcn = @(src, evt) closeApp();
end
