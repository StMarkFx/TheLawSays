import { LinearGradient } from "expo-linear-gradient";
import * as Clipboard from "expo-clipboard";
import React, { useCallback, useMemo, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View
} from "react-native";
import { useFonts, SpaceGrotesk_500Medium, SpaceGrotesk_600SemiBold, SpaceGrotesk_700Bold } from "@expo-google-fonts/space-grotesk";

import { ChatWindow } from "./src/components/ChatWindow";
import { Composer } from "./src/components/Composer";
import { HeaderBar } from "./src/components/HeaderBar";
import { Sidebar } from "./src/components/Sidebar";
import { sendChat, sendFeedback } from "./src/lib/api";
import type { Chunk, Message, Role } from "./src/lib/types";
import { createId } from "./src/lib/utils";
import { colors, fonts } from "./src/theme";

export default function App() {
  const [fontsLoaded] = useFonts({
    SpaceGrotesk_500Medium,
    SpaceGrotesk_600SemiBold,
    SpaceGrotesk_700Bold
  });

  const [messages, setMessages] = useState<Message[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [retrievalUsed, setRetrievalUsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [conversationId, setConversationId] = useState(() => createId("conv"));
  const [error, setError] = useState<string | null>(null);

  const historyPayload = useMemo<Array<{ role: Role; content: string }>>(
    () => messages.map(({ role, content }) => ({ role, content })),
    [messages]
  );

  const handleSubmit = useCallback(
    async (content: string) => {
      if (loading) return;
      setError(null);
      const userMessage: Message = {
        id: createId("msg"),
        role: "user",
        content,
        createdAt: new Date().toISOString()
      };
      setMessages((prev) => [...prev, userMessage]);
      const nextHistory = [...historyPayload, { role: "user" as Role, content }];
      setLoading(true);
      try {
        const response = await sendChat({
          message: content,
          history: nextHistory
        });
        const assistantMessage: Message = {
          id: createId("msg"),
          role: "assistant",
          content: response.answer,
          createdAt: new Date().toISOString()
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setChunks(response.chunks);
        setRetrievalUsed(response.retrieval_used);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      } finally {
        setLoading(false);
      }
    },
    [historyPayload, loading]
  );

  const handleFeedback = useCallback(
    async (rating: "thumbs_up" | "thumbs_down") => {
      if (!conversationId) return;
      try {
        await sendFeedback({
          conversation_id: conversationId,
          rating
        });
      } catch (err) {
        console.warn("Failed to send feedback", err);
      }
    },
    [conversationId]
  );

  const handleCopyAnswer = useCallback(async (content: string) => {
    try {
      await Clipboard.setStringAsync(content);
    } catch (err) {
      setError("Unable to copy answer to clipboard.");
    }
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setChunks([]);
    setRetrievalUsed(false);
    setConversationId(createId("conv"));
    setError(null);
  }, []);

  if (!fontsLoaded) {
    return (
      <SafeAreaView style={styles.loadingScreen}>
        <ActivityIndicator color={colors.accent} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" />
      <LinearGradient
        colors={["#050505", "#0b0c0d", "#101012"]}
        style={styles.gradient}
      >
        <HeaderBar
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          onNewChat={handleNewChat}
        />

        <View style={styles.content}>
          <ChatWindow
            messages={messages}
            loading={loading}
            chunks={retrievalUsed ? chunks : []}
            retrievalUsed={retrievalUsed}
            onFeedback={handleFeedback}
            onCopyAnswer={handleCopyAnswer}
          />
        </View>

        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          keyboardVerticalOffset={Platform.OS === "ios" ? 10 : 0}
        >
          <View style={styles.composerBlock}>
            {error && <Text style={styles.error}>{error}</Text>}
            <Composer onSubmit={handleSubmit} disabled={loading} />
            <Text style={styles.disclaimer}>
              Built for educational research. Consult qualified lawyers for legal advice.
            </Text>
          </View>
        </KeyboardAvoidingView>

        <Sidebar open={isSidebarOpen} onClose={() => setSidebarOpen(false)} loading={loading} />
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background
  },
  gradient: {
    flex: 1
  },
  loadingScreen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background
  },
  content: {
    flex: 1
  },
  composerBlock: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 16
  },
  error: {
    color: colors.danger,
    textAlign: "center",
    marginBottom: 8,
    fontFamily: fonts.regular
  },
  disclaimer: {
    color: colors.muted,
    fontSize: 11,
    textAlign: "center",
    marginTop: 10,
    fontFamily: fonts.regular
  }
});
