import { Feather } from "@expo/vector-icons";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from "react-native";
import Markdown from "react-native-markdown-display";

import type { Chunk, Message } from "../lib/types";
import { colors, fonts } from "../theme";
import { CitationAccordion } from "./CitationAccordion";
import { FeedbackButtons } from "./FeedbackButtons";

interface ChatWindowProps {
  messages: Message[];
  loading: boolean;
  chunks: Chunk[];
  retrievalUsed: boolean;
  onFeedback: (rating: "thumbs_up" | "thumbs_down") => void;
  onCopyAnswer: (content: string) => void;
}

export function ChatWindow({
  messages,
  loading,
  chunks,
  retrievalUsed,
  onFeedback,
  onCopyAnswer
}: ChatWindowProps) {
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const scrollRef = useRef<ScrollView>(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  const assistantMessages = useMemo(
    () => messages.filter((message) => message.role === "assistant"),
    [messages]
  );
  const latestAssistant = assistantMessages[assistantMessages.length - 1];
  const showEmptyState = messages.length === 0;

  useEffect(() => {
    setFeedbackGiven(false);
  }, [latestAssistant?.id]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollToEnd({ animated: true });
    }
  }, [messages, loading]);

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 450,
      useNativeDriver: true
    }).start();
  }, [fadeAnim]);

  const handleFeedback = (rating: "thumbs_up" | "thumbs_down") => {
    onFeedback(rating);
    setFeedbackGiven(true);
  };

  return (
    <ScrollView
      ref={scrollRef}
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      {showEmptyState ? (
        <Animated.View style={[styles.emptyState, { opacity: fadeAnim }]}> 
          {loading ? (
            <ActivityIndicator color={colors.accent} />
          ) : (
            <>
              <Text style={styles.emptyTitle}>
                Empower yourself with Nigerian law at your fingertips.
              </Text>
              <Text style={styles.emptyBody}>
                Ask any question about Federal or Lagos State statutes and get instant answers backed by exact
                sections and citations.
              </Text>
            </>
          )}
        </Animated.View>
      ) : (
        <View style={styles.messages}>
          {messages.map((message) => {
            const isUser = message.role === "user";
            const isLatestAssistant = !isUser && message.id === latestAssistant?.id;

            if (isUser) {
              return (
                <View key={message.id} style={[styles.userRow, styles.messageBlock]}>
                  <View style={styles.userBubble}>
                    <Text style={styles.userText}>{message.content}</Text>
                  </View>
                </View>
              );
            }

            return (
              <View key={message.id} style={[styles.assistantRow, styles.messageBlock]}>
                <View style={styles.assistantBubble}>
                  <Markdown style={markdownStyles}>{message.content}</Markdown>
                  <View style={styles.metaRow}>
                    <TouchableOpacity
                      onPress={() => onCopyAnswer(message.content)}
                      style={styles.metaButton}
                      accessibilityLabel="Copy answer"
                    >
                      <Feather name="copy" size={16} color={colors.muted} />
                    </TouchableOpacity>
                    <FeedbackButtons onFeedback={handleFeedback} disabled={feedbackGiven} />
                  </View>
                  {isLatestAssistant && retrievalUsed && chunks.length > 0 && (
                    <View style={styles.citationBlock}>
                      <CitationAccordion chunks={chunks} />
                    </View>
                  )}
                </View>
              </View>
            );
          })}

          {loading && (
            <View style={styles.loadingRow}>
              <ActivityIndicator size="small" color={colors.accent} />
              <Text style={styles.loadingText}>Thinking...</Text>
            </View>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const markdownStyles = {
  body: {
    color: colors.foreground,
    fontSize: 16,
    lineHeight: 24,
    fontFamily: fonts.regular
  },
  strong: {
    fontFamily: fonts.medium
  },
  blockquote: {
    borderLeftWidth: 2,
    borderLeftColor: colors.quoteBorder,
    paddingLeft: 12,
    color: colors.muted
  },
  paragraph: {
    marginTop: 0,
    marginBottom: 12
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 16
  },
  content: {
    paddingTop: 18,
    paddingBottom: 24
  },
  emptyState: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 16
  },
  emptyTitle: {
    color: colors.foreground,
    fontSize: 22,
    textAlign: "center",
    fontFamily: fonts.medium,
    marginBottom: 12
  },
  emptyBody: {
    color: colors.muted,
    fontSize: 16,
    lineHeight: 22,
    textAlign: "center",
    fontFamily: fonts.regular
  },
  messages: {},
  messageBlock: {
    marginBottom: 18
  },
  userRow: {
    alignItems: "flex-end"
  },
  userBubble: {
    maxWidth: "80%",
    backgroundColor: colors.bubbleUser,
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 22
  },
  userText: {
    color: colors.foreground,
    fontSize: 16,
    lineHeight: 22,
    fontFamily: fonts.regular
  },
  assistantRow: {
    alignItems: "flex-start"
  },
  assistantBubble: {
    maxWidth: "90%"
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 12
  },
  metaButton: {
    height: 34,
    width: 34,
    borderRadius: 17,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12
  },
  citationBlock: {
    marginTop: 12
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center"
  },
  loadingText: {
    color: colors.muted,
    fontSize: 14,
    fontFamily: fonts.regular,
    marginLeft: 8
  }
});
