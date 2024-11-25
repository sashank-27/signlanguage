window.addEventListener('DOMContentLoaded', () => {
    const videoStream = document.getElementById('video-stream');

    function fetchStream() {
        videoStream.src = "/api/video_feed";
    }

    fetchStream();
});
