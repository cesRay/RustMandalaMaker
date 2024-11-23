use opencv::prelude::*;
use opencv::imgcodecs;
use opencv::highgui;
use opencv::core;
use opencv::imgproc;
use rfd::FileDialog;
// use statrs::statistics::Statistics;

fn main() -> opencv::Result<()> {
    // Make user select the file in the file manager
    let file_path = FileDialog::new()
        .add_filter("Image files", &["jpg", "png", "bmp", "jpeg"])
        .pick_file()
        .expect("No file selected");

    // Read selected image.
    let mut img = imgcodecs::imread(file_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;

    // Check if the image was read correctly
    if img.empty() {
        panic!("Could not read the image");
    }

    // Convert the image to grayscale
    let mut gray = core::Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Apply the Canny edge detector
    let mut edges = core::Mat::default();
    imgproc::canny(&gray, &mut edges, 100.0, 200.0, 3, false)?;

    let mut edges_bgr = core::Mat::default();
    imgproc::cvt_color(&edges, &mut edges_bgr, imgproc::COLOR_GRAY2BGR, 0)?;

    // Make all white pixels green in the edges image
    for y in 0..edges_bgr.rows() {
        for x in 0..edges_bgr.cols() {
            let pixel = edges_bgr.at_2d_mut::<core::Vec3b>(y, x)?;
            if pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255 {
                *pixel = core::Vec3b::from([0, 255, 0]); // Green color in BGR format
            }
        }
    }

    // Reflect image in the vertical axis
    let mut reflected_y = core::Mat::default();
    core::flip(&edges_bgr, &mut reflected_y, 0)?;

    // Reflect image in the horizontal axis
    let mut reflected_x = core::Mat::default();
    core::flip(&edges_bgr, &mut reflected_x, 1)?;

    // Reflect image in both axes
    let mut reflected_xy = core::Mat::default();
    core::flip(&edges_bgr, &mut reflected_xy, -1)?;

    // Concatenate the images in x
    let mut concatenated_x = core::Mat::default();
    core::hconcat(&core::Vector::<core::Mat>::from(vec![edges_bgr, reflected_x]), &mut concatenated_x)?;


    // Concatenate the images in y
    let mut concatenated_y = core::Mat::default();
    core::hconcat(&core::Vector::<core::Mat>::from(vec![reflected_y, reflected_xy]), &mut concatenated_y)?;

    // Concatenate the images vertically
    let mut finished = core::Mat::default();
    core::vconcat(&core::Vector::<core::Mat>::from(vec![concatenated_x, concatenated_y]), &mut finished)?;

    // Display the image in a window
    highgui::imshow("image", &finished)?;
    highgui::wait_key(0)?;

    // Save the image to a file
    imgcodecs::imwrite("public/images/edges.jpg", &edges, &core::Vector::<i32>::new())?;

    Ok(())
}
