// JavaScript code for the feedback form in the Footer
// Handles form validation when the user attempts to submit the form and resets the form inputs after the form submission

document.getElementById('feedback-form').addEventListener('submit', function(event) {
  const feedback = document.getElementsByName('feedback')[0].value;

  // Display alert message if no feedback is given when submitting the feedback form
  // trim() ensures even blank spaces in the form do not get accepted
  if (!feedback.trim())
  {
    // Don't allow form to be submitted
    event.preventDefault();
    alert('Please type your feedback before submitting.');  
  } 
  else
    alert('We have received your feedback!')

    // Reset feedback form input field after submission
    const feedbackForm = document.getElementById('feedback-form');
    feedbackForm.reset();
});