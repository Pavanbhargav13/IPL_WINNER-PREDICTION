import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc, query, orderBy, getDocs, limit, serverTimestamp } from 'firebase/firestore';

// ============================================================================
// FIREBASE CONFIGURATION
// To activate AI Memory, replace these placeholders with your actual Firebase 
// project credentials from the Firebase Console (Project Settings > General).
// ============================================================================
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "your-app.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-app.appspot.com",
  messagingSenderId: "YOUR_MESSAGING_ID",
  appId: "YOUR_APP_ID"
};

let db = null;
let isFirebaseEnabled = false;

try {
  // Check if user has actually replaced the placeholder keys
  if (firebaseConfig.apiKey !== "YOUR_API_KEY") {
    const app = initializeApp(firebaseConfig);
    db = getFirestore(app);
    isFirebaseEnabled = true;
    console.log("🔥 Firebase AI Memory Initialized Successfully!");
  } else {
    console.warn("⚠️ Firebase keys not found. Using local browser memory fallback for AI Chatbot.");
  }
} catch (error) {
  console.error("Firebase initialization error:", error);
}

/**
 * Saves a new conversation memory to Firebase (or localStorage fallback).
 * @param {string} role 'user' or 'ai'
 * @param {string} content The message content
 */
export const saveAiMemory = async (role, content) => {
  if (isFirebaseEnabled && db) {
    try {
      await addDoc(collection(db, "ai_memories"), {
        role,
        content,
        timestamp: serverTimestamp()
      });
    } catch (e) {
      console.error("Error saving memory to Firebase:", e);
    }
  } else {
    // Fallback: Use LocalStorage if Firebase isn't configured yet
    const localMemories = JSON.parse(localStorage.getItem('ai_memories') || '[]');
    localMemories.push({ role, content, timestamp: Date.now() });
    localStorage.setItem('ai_memories', JSON.stringify(localMemories));
  }
};

/**
 * Retrieves the last N conversation memories from Firebase (or localStorage).
 * @param {number} maxRecords Number of memories to fetch
 */
export const fetchAiMemories = async (maxRecords = 10) => {
  if (isFirebaseEnabled && db) {
    try {
      const q = query(collection(db, "ai_memories"), orderBy("timestamp", "desc"), limit(maxRecords));
      const querySnapshot = await getDocs(q);
      const memories = [];
      querySnapshot.forEach((doc) => {
        memories.push(doc.data());
      });
      return memories.reverse(); // Return in chronological order
    } catch (e) {
      console.error("Error fetching memory from Firebase:", e);
      return [];
    }
  } else {
    // Fallback: LocalStorage
    const localMemories = JSON.parse(localStorage.getItem('ai_memories') || '[]');
    return localMemories.slice(-maxRecords);
  }
};
